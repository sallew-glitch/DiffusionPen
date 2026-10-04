"""Evaluate DiffusionPen on legibility (CER via TrOCR) and style fidelity (HWD).

Protocol: pick IAM writers (unseen test writers by default), take 5 of each writer's
word images as style references, then generate other words that same writer wrote.
The generated words are scored against the writer's real images of the same words.

Example (compare samplers; the writers, references and initial noise are identical across runs):
    python eval.py --scheduler ddim --sampling_steps 50
    python eval.py --scheduler dpm --sampling_steps 20

Outputs in --out_dir:
    protocol.json           writers, reference images and target words (reused by later runs)
    refs/<writer>/          the 5 style references fed to the model
    real/<writer>/          real images of the target words (+ transcriptions.json)
    real_holdout/<writer>/  other real words by the same writer, for the real-vs-real HWD reference
    fake_<tag>/<writer>/    generated target words (+ transcriptions.json)
    results_<tag>.json      scores of one run; a comparison table of all runs is printed at the end
"""
import argparse
import glob
import json
import os
import random
import shutil
import time
from collections import defaultdict
from pathlib import Path

import numpy as np
import torch
from PIL import Image, ImageOps
from torch.nn import DataParallel
from torchvision import transforms
from diffusers import AutoencoderKL, DDIMScheduler, DPMSolverMultistepScheduler
from transformers import CanineModel, CanineTokenizer

from unet import UNetModel
from feature_extractor import ImageEncoder
from utils.auxilary_functions import image_resize_PIL, centered_PIL
from train import crop_whitespace_width

IAM_ROOT = './iam_data/words'
SPLITS = {
    'test': ('./utils/splits_words/iam_test.txt', 'unseen writers'),
    'train': ('./utils/splits_words/iam_train_val.txt', 'writers seen in training'),
}
#same character set as train.py; words with other characters are skipped
VOCAB = set('!"#&\'()*+,-./0123456789:;?ABCDEFGHIJKLMNOPQRSTUVWXYZabcdefghijklmnopqrstuvwxyz')
NUM_REFS = 5
STYLE_CLASSES = 339


############################ PROTOCOL ############################
def build_protocol(args):
    split_file, _ = SPLITS[args.split]
    by_writer = defaultdict(list)
    with open(split_file, 'r') as f:
        for line in f:
            parts = line.rstrip('\n').split(',', 2)
            if len(parts) != 3:
                continue
            path, writer, word = parts
            #at least one letter: punctuation-only "words" like '...' only add OCR noise
            if len(word) >= 3 and set(word) <= VOCAB and any(c.isalpha() for c in word) and os.path.exists(os.path.join(IAM_ROOT, path)):
                by_writer[writer].append((path, word))

    rng = random.Random(args.seed)
    needed = NUM_REFS + 2 * args.words_per_writer
    eligible = sorted(w for w, items in by_writer.items() if len(items) >= needed and sum(len(word) > 3 for _, word in items) >= NUM_REFS)
    if len(eligible) < args.num_writers:
        raise SystemExit(f'Only {len(eligible)} writers have {needed} usable words; lower --num_writers or --words_per_writer')

    writers = {}
    for writer in sorted(rng.sample(eligible, args.num_writers)):
        items = sorted(by_writer[writer])
        rng.shuffle(items)
        #references need more than 3 letters, like the style images picked in train.py
        refs = [it for it in items if len(it[1]) > 3][:NUM_REFS]
        rest = [it for it in items if it not in refs]
        k = args.words_per_writer
        writers[writer] = {
            'refs': [p for p, _ in refs],
            'targets': [{'path': p, 'text': t} for p, t in rest[:k]],
            'holdout': [{'path': p, 'text': t} for p, t in rest[k:2 * k]],
        }
    return {'split': args.split, 'seed': args.seed, 'words_per_writer': args.words_per_writer, 'writers': writers}


def load_or_create_protocol(args):
    path = os.path.join(args.out_dir, 'protocol.json')
    if os.path.exists(path):
        with open(path, 'r') as f:
            protocol = json.load(f)
        print(f'Reusing {path} ({len(protocol["writers"])} writers, split={protocol["split"]}); '
              f'--split/--num_writers/--words_per_writer/--seed are ignored. Use a new --out_dir to change them.')
        return protocol

    protocol = build_protocol(args)
    os.makedirs(args.out_dir, exist_ok=True)
    with open(path, 'w') as f:
        json.dump(protocol, f, indent=2)

    #copy the real images so every folder has the <writer>/<image> layout the hwd package expects
    for name in ['real', 'real_holdout']:
        key = 'targets' if name == 'real' else 'holdout'
        transcriptions = {}
        for writer, entry in protocol['writers'].items():
            os.makedirs(os.path.join(args.out_dir, name, writer), exist_ok=True)
            for i, item in enumerate(entry[key]):
                rel = str(Path(writer) / f'{i:03d}.png')
                shutil.copy(os.path.join(IAM_ROOT, item['path']), os.path.join(args.out_dir, name, rel))
                transcriptions[rel] = item['text']
        with open(os.path.join(args.out_dir, name, 'transcriptions.json'), 'w') as f:
            json.dump(transcriptions, f, indent=2)
    for writer, entry in protocol['writers'].items():
        os.makedirs(os.path.join(args.out_dir, 'refs', writer), exist_ok=True)
        for i, ref in enumerate(entry['refs']):
            shutil.copy(os.path.join(IAM_ROOT, ref), os.path.join(args.out_dir, 'refs', writer, f'{i}.png'))
    print(f'Created protocol: {len(protocol["writers"])} writers ({SPLITS[args.split][1]}), '
          f'{args.words_per_writer} words each -> {args.out_dir}')
    return protocol


############################ GENERATION ############################
def load_models(args):
    device_ids = [int(''.join(filter(str.isdigit, args.device)) or 0)] if args.device.startswith('cuda') else None

    tokenizer = CanineTokenizer.from_pretrained("google/canine-c")
    text_encoder = CanineModel.from_pretrained("google/canine-c")
    text_encoder = DataParallel(text_encoder, device_ids=device_ids).to(args.device)

    #UNetModel reads these two from args
    args.interpolation, args.mix_rate = False, None
    unet = UNetModel(image_size=(64, 256), in_channels=4, model_channels=320, out_channels=4, num_res_blocks=1, attention_resolutions=(1, 1), channel_mult=(1, 1), num_heads=4, num_classes=STYLE_CLASSES, context_dim=320, vocab_size=80, text_encoder=text_encoder, args=args)
    unet = DataParallel(unet, device_ids=device_ids).to(args.device)
    unet.load_state_dict(torch.load(f'{args.save_path}/models/ema_ckpt.pt', map_location=args.device))
    unet.eval()

    vae = AutoencoderKL.from_pretrained(args.stable_dif_path, subfolder="vae")
    vae = DataParallel(vae, device_ids=device_ids).to(args.device)
    vae.requires_grad_(False)

    if args.scheduler == 'dpm':
        noise_scheduler = DPMSolverMultistepScheduler.from_pretrained(args.stable_dif_path, subfolder="scheduler", algorithm_type="dpmsolver++", solver_order=2)
    else:
        noise_scheduler = DDIMScheduler.from_pretrained(args.stable_dif_path, subfolder="scheduler")

    feature_extractor = ImageEncoder(model_name='mobilenetv2_100', num_classes=0, pretrained=True, trainable=True)
    state_dict = torch.load(args.style_path, map_location=args.device)
    model_dict = feature_extractor.state_dict()
    model_dict.update({k: v for k, v in state_dict.items() if k in model_dict and model_dict[k].shape == v.shape})
    feature_extractor.load_state_dict(model_dict)
    feature_extractor = DataParallel(feature_extractor, device_ids=device_ids).to(args.device)
    feature_extractor.requires_grad_(False)
    feature_extractor.eval()

    return unet, vae, noise_scheduler, feature_extractor, tokenizer


def load_style_image(path, transform):
    #same preprocessing as the style images in Diffusion.sampling (train.py)
    img = Image.open(path).convert('RGB')
    img = img.resize((int(img.width * 64 / img.height), 64))
    if img.width < 256:
        img = ImageOps.pad(img, size=(256, 64), color="white")
    else:
        while img.width > 256:
            img = image_resize_PIL(img, width=img.width - 20)
        img = centered_PIL(img, (64, 256), border_value=255.0)
    return transform(img)


@torch.no_grad()
def generate_words(texts, style_features, unet, vae, noise_scheduler, tokenizer, args, generator):
    n = len(texts)
    #the UNet averages 5 style vectors per sample, so repeat the writer's 5 features for every word
    style = style_features.repeat(n, 1)
    text_features = tokenizer(texts, padding="max_length", truncation=True, return_tensors="pt", max_length=40).to(args.device)
    x = torch.randn((n, 4, 8, 32), generator=generator).to(args.device)
    noise_scheduler.set_timesteps(args.sampling_steps)
    for t in noise_scheduler.timesteps:
        timesteps = torch.full((n,), t.item(), dtype=torch.long, device=args.device)
        noise_pred = unet(x, timesteps, text_features, None, style_extractor=style)
        x = noise_scheduler.step(noise_pred, t, x).prev_sample
    images = vae.module.decode(x / 0.18215).sample
    images = (images / 2 + 0.5).clamp(0, 1).cpu()
    return [transforms.ToPILImage()(img).convert('L') for img in images]


def crop_word(img):
    #crop to the ink so generated words are framed like the real IAM crops
    try:
        return Image.fromarray(crop_whitespace_width(img))
    except Exception:
        return img


def run_generation(protocol, args):
    fake_dir = os.path.join(args.out_dir, f'fake_{args.tag}')
    print(f'Loading models, sampler={args.scheduler}, steps={args.sampling_steps}, device={args.device}')
    unet, vae, noise_scheduler, feature_extractor, tokenizer = load_models(args)
    transform = transforms.Compose([transforms.ToTensor(), transforms.Normalize((0.5, 0.5, 0.5), (0.5, 0.5, 0.5))])

    transcriptions = {}
    num_words = 0
    start = time.time()
    for w_idx, (writer, entry) in enumerate(protocol['writers'].items()):
        refs = torch.stack([load_style_image(os.path.join(IAM_ROOT, p), transform) for p in entry['refs']]).to(args.device)
        style_features = feature_extractor(refs)
        #fixed noise per writer, so different samplers start from the same latents
        generator = torch.Generator().manual_seed(protocol['seed'] * 1000 + w_idx)
        os.makedirs(os.path.join(fake_dir, writer), exist_ok=True)

        texts = [t['text'] for t in entry['targets']]
        for b in range(0, len(texts), args.batch_size):
            batch = texts[b:b + args.batch_size]
            images = generate_words(batch, style_features, unet, vae, noise_scheduler, tokenizer, args, generator)
            for i, (img, text) in enumerate(zip(images, batch)):
                rel = str(Path(writer) / f'{b + i:03d}.png')
                crop_word(img).save(os.path.join(fake_dir, rel))
                transcriptions[rel] = text
        num_words += len(texts)
        print(f'[{w_idx + 1}/{len(protocol["writers"])}] writer {writer}: {len(texts)} words, {time.time() - start:.0f}s elapsed')

    with open(os.path.join(fake_dir, 'transcriptions.json'), 'w') as f:
        json.dump(transcriptions, f, indent=2)
    elapsed = time.time() - start
    return {'num_words': num_words, 'gen_seconds': round(elapsed, 1), 'sec_per_word': round(elapsed / num_words, 3)}


############################ SCORING ############################
def legibility(dataset, cer_score):
    from torchmetrics.text import CharErrorRate
    preds, labels, authors = cer_score.digest(dataset)
    preds = [p.strip() for p in preds]
    cer = CharErrorRate()(preds, labels).item()
    #TrOCR is a line model and often appends ' .' to single words, so the normalized scores ignore case/punctuation/spaces
    norm = lambda s: ''.join(c for c in s.lower() if c.isalnum())
    preds_n, labels_n = [norm(p) for p in preds], [norm(l) for l in labels]
    cer_norm = CharErrorRate()(preds_n, labels_n).item()
    word_acc = float(np.mean([p == l for p, l in zip(preds_n, labels_n)]))
    samples = [{'author': a, 'text': l, 'pred': p} for a, l, p in zip(authors, labels, preds)]
    return {'cer': round(cer, 4), 'cer_norm': round(cer_norm, 4), 'word_acc': round(word_acc, 4)}, samples


def score(args, fake_dir):
    from hwd.datasets import FolderDataset
    from hwd.scores import HWDScore, CERScore, FIDScore

    print('Scoring: HWD (style) and TrOCR CER (legibility)...')
    real = lambda: FolderDataset(os.path.join(args.out_dir, 'real'))
    fake = lambda: FolderDataset(fake_dir)
    hwd_score = HWDScore(height=32)
    cer_score = CERScore(height=64, path=args.htr_model)

    results = {'hwd': round(hwd_score(fake(), real()), 4)}
    fake_leg, samples = legibility(fake(), cer_score)
    results.update(fake_leg)
    if args.fid:
        results['fid'] = round(FIDScore(height=32)(fake(), real()), 2)

    #reference values from real handwriting, computed once per protocol
    baseline_path = os.path.join(args.out_dir, 'real_baseline.json')
    if os.path.exists(baseline_path):
        with open(baseline_path, 'r') as f:
            baseline = json.load(f)
    else:
        print('Scoring real images once for reference values...')
        baseline, real_samples = legibility(real(), cer_score)
        baseline['hwd'] = round(hwd_score(FolderDataset(os.path.join(args.out_dir, 'real_holdout')), real()), 4)
        with open(baseline_path, 'w') as f:
            json.dump(baseline, f, indent=2)
        with open(os.path.join(args.out_dir, 'real', 'ocr_predictions.json'), 'w') as f:
            json.dump(real_samples, f, indent=2)
    return results, baseline, samples


def print_table(out_dir, baseline):
    rows = []
    for path in sorted(glob.glob(os.path.join(out_dir, 'results_*.json'))):
        with open(path, 'r') as f:
            rows.append(json.load(f))
    header = f'{"run":<14}{"HWD (style)":>13}{"CER":>8}{"CER norm":>12}{"word acc":>10}{"sec/word":>10}'
    print('\n' + header + '\n' + '-' * len(header))
    for r in rows:
        print(f'{r["tag"]:<14}{r["hwd"]:>13.4f}{r["cer"]:>8.4f}{r["cer_norm"]:>12.4f}{r["word_acc"]:>10.3f}'
              f'{(r.get("sec_per_word") or float("nan")):>10.2f}')
    print(f'{"real (ref.)":<14}{baseline["hwd"]:>13.4f}{baseline["cer"]:>8.4f}{baseline["cer_norm"]:>12.4f}{baseline["word_acc"]:>10.3f}{"":>10}')
    print('Lower is better for HWD and CER; higher is better for word acc. "real" row: HWD between two disjoint sets of\n'
          'each writer\'s real words (reference level for "same writer"), CER of TrOCR on the real target words (the OCR ceiling).')


def main():
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument('--out_dir', type=str, default='./eval_runs/iam_test', help='one folder per protocol; runs inside it are directly comparable')
    parser.add_argument('--split', type=str, default='test', choices=list(SPLITS), help='test = unseen writers (few-shot), train = writers seen in training')
    parser.add_argument('--num_writers', type=int, default=10)
    parser.add_argument('--words_per_writer', type=int, default=20)
    parser.add_argument('--seed', type=int, default=0)
    parser.add_argument('--scheduler', type=str, default='dpm', choices=['ddim', 'dpm'])
    parser.add_argument('--sampling_steps', type=int, default=None, help='default: 50 for ddim, 20 for dpm')
    parser.add_argument('--batch_size', type=int, default=16, help='words generated per diffusion pass')
    parser.add_argument('--tag', type=str, default=None, help='run name (default: <scheduler><steps>)')
    parser.add_argument('--skip_generation', action='store_true', help='only re-score an existing fake_<tag> folder')
    parser.add_argument('--fid', action='store_true', help='also compute FID (needs many images to be meaningful)')
    parser.add_argument('--htr_model', type=str, default='microsoft/trocr-base-handwritten', help='TrOCR model used for CER')
    parser.add_argument('--save_path', type=str, default='./diffusionpen_iam_model_path')
    parser.add_argument('--style_path', type=str, default='./style_models/iam_style_diffusionpen.pth')
    parser.add_argument('--stable_dif_path', type=str, default='./stable-diffusion-v1-5')
    parser.add_argument('--device', type=str, default='cuda:0' if torch.cuda.is_available() else 'cpu')
    args = parser.parse_args()
    if args.sampling_steps is None:
        args.sampling_steps = 50 if args.scheduler == 'ddim' else 20
    if args.tag is None:
        args.tag = f'{args.scheduler}{args.sampling_steps}'

    protocol = load_or_create_protocol(args)
    fake_dir = os.path.join(args.out_dir, f'fake_{args.tag}')
    results_path = os.path.join(args.out_dir, f'results_{args.tag}.json')

    timing = {}
    if args.skip_generation:
        if not os.path.exists(os.path.join(fake_dir, 'transcriptions.json')):
            raise SystemExit(f'{fake_dir} has no generated images to score')
        if os.path.exists(results_path):
            with open(results_path, 'r') as f:
                old = json.load(f)
            timing = {k: old[k] for k in ['num_words', 'gen_seconds', 'sec_per_word'] if k in old}
    else:
        timing = run_generation(protocol, args)

    results, baseline, samples = score(args, fake_dir)
    results = {'tag': args.tag, 'scheduler': args.scheduler, 'sampling_steps': args.sampling_steps, **results, **timing}
    with open(results_path, 'w') as f:
        json.dump(results, f, indent=2)
    with open(os.path.join(fake_dir, 'ocr_predictions.json'), 'w') as f:
        json.dump(samples, f, indent=2)
    print_table(args.out_dir, baseline)


if __name__ == '__main__':
    main()
