"""Generate ms_fusion structure diagram as PNG."""
import matplotlib.pyplot as plt
from matplotlib.patches import FancyBboxPatch, FancyArrowPatch

fig, ax = plt.subplots(figsize=(17, 12))
ax.set_xlim(0, 17)
ax.set_ylim(-1.5, 12)
ax.axis('off')

C_INPUT = '#dae8fc'; C_INPUT_E = '#6c8ebf'
C_DEC = '#f5f5f5'; C_DEC_E = '#666666'
C_COLLECT = '#fff2cc'; C_COLLECT_E = '#d6b656'
C_LAST = '#d5e8d4'; C_LAST_E = '#82b366'
C_GATE = '#e1d5e7'; C_GATE_E = '#9673a6'
C_FUSE = '#fad7ac'; C_FUSE_E = '#b46504'
C_OUT = '#f8cecc'; C_OUT_E = '#b85450'

def box(x, y, w, h, text, fc, ec, fs=9, fw='normal', alpha=1.0):
    p = FancyBboxPatch((x, y), w, h, boxstyle="round,pad=0.05",
                       linewidth=1.5, edgecolor=ec, facecolor=fc, alpha=alpha)
    ax.add_patch(p)
    if text:
        ax.text(x + w/2, y + h/2, text, ha='center', va='center',
                fontsize=fs, fontweight=fw)

def arrow(x1, y1, x2, y2, color='black', style='-', lw=1.2):
    a = FancyArrowPatch((x1, y1), (x2, y2),
                        arrowstyle='->', mutation_scale=14,
                        color=color, linestyle=style, linewidth=lw,
                        shrinkA=2, shrinkB=2)
    ax.add_patch(a)

ax.text(8.5, 11.6, 'C1 — MultiScaleFusion (ms_fusion) data flow',
        ha='center', fontsize=17, fontweight='bold')
ax.text(8.5, 11.15, '36 decoder blocks; collect features at layers 8/17/26/34; '
                    'softmax-gated weighted sum; concat with last (layer 35) output.',
        ha='center', fontsize=10.5, style='italic', color='#444')

dec_y = 10.0
dec_h = 0.65
box(0.2, dec_y, 1.6, dec_h, 'hidden in\n(B,N,hw,1024)', C_INPUT, C_INPUT_E, fs=8.5, fw='bold')

dec_layout = [
    (2.0, 1.4, 'Block 0', C_DEC, C_DEC_E, False),
    (3.5, 0.4, '...', None, None, False),
    (4.0, 1.6, 'Block 8\n(f8)', C_COLLECT, C_COLLECT_E, True),
    (5.7, 0.4, '...', None, None, False),
    (6.2, 1.6, 'Block 17\n(f17)', C_COLLECT, C_COLLECT_E, True),
    (7.9, 0.4, '...', None, None, False),
    (8.4, 1.6, 'Block 26\n(f26)', C_COLLECT, C_COLLECT_E, True),
    (10.1, 0.4, '...', None, None, False),
    (10.6, 1.6, 'Block 34\n(f34)', C_COLLECT, C_COLLECT_E, True),
    (12.4, 1.9, 'Block 35 (last)\n→ last_hidden', C_LAST, C_LAST_E, True),
]
prev_x_end = 1.8
for x, w, txt, fc, ec, _ in dec_layout:
    if fc is None:
        ax.text(x + w/2, dec_y + dec_h/2, txt, ha='center', va='center',
                fontsize=18, fontweight='bold')
    else:
        box(x, dec_y, w, dec_h, txt, fc, ec, fs=8.5, fw='bold')
    arrow(prev_x_end, dec_y + dec_h/2, x, dec_y + dec_h/2)
    prev_x_end = x + w

dec8_cx = 4.0 + 1.6/2
dec17_cx = 6.2 + 1.6/2
dec26_cx = 8.4 + 1.6/2
dec34_cx = 10.6 + 1.6/2
dec35_cx = 12.4 + 1.9/2

box(0.2, 1.5, 14.6, 7.5, '', C_FUSE, '#e8a050', fs=10, alpha=0.15)
ax.text(7.5, 8.65, 'MultiScaleFusion (ms_fusion)',
        ha='center', fontsize=14, fontweight='bold', color=C_FUSE_E)

f_x = 0.7; f_w = 1.7; f_h = 0.65
fy_specs = [
    (dec8_cx, 7.6, 'f8\n(B*N, hw, 1024)'),
    (dec17_cx, 6.7, 'f17\n(B*N, hw, 1024)'),
    (dec26_cx, 5.8, 'f26\n(B*N, hw, 1024)'),
    (dec34_cx, 4.9, 'f34\n(B*N, hw, 1024)'),
]
f_centers = []
for src_cx, fy, txt in fy_specs:
    box(f_x, fy, f_w, f_h, txt, C_COLLECT, C_COLLECT_E, fs=9)
    arrow(src_cx, dec_y, src_cx, fy + f_h + 0.05, color=C_COLLECT_E, style='--', lw=1.0)
    arrow(src_cx, fy + f_h, f_x + f_w/2, fy + f_h, color=C_COLLECT_E, style='--', lw=1.0)
    f_centers.append((f_x + f_w, fy + f_h/2))

gate_x = 4.5; gate_w = 2.8
box(gate_x, 7.4, gate_w, 0.9, 'gate_logits (Param)\ninit = [−5, −5, −5, +5]   shape: (4,)',
    C_GATE, C_GATE_E, fs=9.5, fw='bold')
box(gate_x + 0.4, 6.3, gate_w - 0.8, 0.65, 'softmax(dim=0)',
    C_INPUT, C_INPUT_E, fs=10, fw='bold')
box(gate_x, 5.0, gate_w, 0.95, 'weights = softmax(gate_logits)\n[w8, w17, w26, w34]\ninit ≈ [~0, ~0, ~0, ~1]',
    C_INPUT, C_INPUT_E, fs=9.5)
arrow(gate_x + gate_w/2, 7.4, gate_x + gate_w/2, 6.95, color=C_GATE_E)
arrow(gate_x + gate_w/2, 6.3, gate_x + gate_w/2, 5.95, color=C_INPUT_E)

ws_x = 8.5; ws_y = 5.6; ws_w = 2.4; ws_h = 1.4
box(ws_x, ws_y, ws_w, ws_h, 'weighted sum\n\nΣ wᵢ · fᵢ', C_LAST, C_LAST_E, fs=12, fw='bold')

for fx_end, fy_c in f_centers:
    arrow(fx_end, fy_c, ws_x, ws_y + ws_h/2, color='black', lw=1.0)

arrow(gate_x + gate_w, 5.45, ws_x, ws_y + 0.3, color=C_GATE_E, style='--', lw=1.2)
ax.text((gate_x + gate_w + ws_x)/2, 5.25, 'weights', fontsize=8, color=C_GATE_E,
        ha='center', style='italic')

box(12.0, 5.95, 2.4, 0.7, 'fused\n(B*N, hw, 1024)', C_LAST, C_LAST_E, fs=10, fw='bold')
arrow(ws_x + ws_w, ws_y + ws_h/2, 12.0, 6.3, color='black')

box(12.0, 3.5, 2.4, 0.8, 'last_hidden\n(layer 35)\n(B*N, hw, 1024)', C_LAST, C_LAST_E, fs=9.5, fw='bold')
arrow(dec35_cx, dec_y, dec35_cx, 4.3 + 0.05, color=C_LAST_E, style='--', lw=1.0)
arrow(dec35_cx, 4.3, 14.4, 4.3, color=C_LAST_E, style='--', lw=1.0)

cc_x = 8.5; cc_y = 2.4; cc_w = 2.4; cc_h = 0.75
box(cc_x, cc_y, cc_w, cc_h, 'Concat (dim = −1)', C_OUT, C_OUT_E, fs=11, fw='bold')
arrow(12.0, 6.0, cc_x + cc_w, cc_y + 0.6, color='black')
arrow(12.0, 3.7, cc_x + cc_w, cc_y + 0.3, color='black')

box(2.5, 2.3, 4.2, 0.95,
    'encoded output\n(B*N, hw, 2048)\n→ point_decoder / camera path',
    C_OUT, C_OUT_E, fs=10, fw='bold')
arrow(cc_x, cc_y + cc_h/2, 6.7, 2.77, color='black')

note = ('Notes:\n'
        '• Decoder: 36 blocks (idx 0..35), embed dim 1024.\n'
        '• ms_collect_indices = (8, 17, 26, 34) — 4 evenly spaced layers.\n'
        '• gate_logits init = [−5,−5,−5,+5] → softmax ≈ [~0,~0,~0,~1] → init behaves as "only f34".\n'
        '• During training, gate_logits are learned → fusion mixes earlier-layer features.\n'
        '• Final encoder output = concat(fused, last_hidden) → 2048-dim, fed to point head & camera head.\n'
        '• ablate_msfusion=True: skip ms_fusion, concat (last-2, last) like vanilla Pi3.')
ax.text(0.2, 0.9, note, fontsize=9, va='top', ha='left',
        bbox=dict(boxstyle='round,pad=0.45', facecolor='#f5f5f5', edgecolor='#888'))

plt.tight_layout()
out = '/home/wangqw/video_program/Pi3/ms_fusion_C1.png'
plt.savefig(out, dpi=160, bbox_inches='tight', facecolor='white')
print(f'Saved: {out}')
