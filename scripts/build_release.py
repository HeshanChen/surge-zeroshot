"""Build the publication repository staging tree from the working repo.
Copies ONLY the manifest below (the certified-correct versions at current HEAD), so the
release contains no scratch code, no dead experiments, no working notes.
Usage: python scripts/build_release.py [dest]   (default ~/Desktop/surge-paper-release)
If dest is an existing git clone (has .git), files are added or overwritten in place and nothing else is
removed; a non-git dest is rebuilt from scratch.
September 2026 revision: continuous-window evaluations, symmetric GTSM head-to-head, block bootstrap, clean factor
study (v7e attention model, Chronos), Europe corpus intervention, fixed-count composition ablation, matched-update
control, GEFS re-issue with peaks inside the window.
"""
import os, shutil, sys

SRC = '/Users/heshan/Desktop/surge_fm'
DST = sys.argv[1] if len(sys.argv) > 1 else os.path.expanduser('~/Desktop/surge-paper-release')

MANIFEST = {
  # ---- library code ----
  'src/data/clean_surge.py': 'src/data/clean_surge.py',
  'src/data/dataset_v0.py': 'src/data/dataset_v0.py',
  'src/data/__init__.py': 'src/data/__init__.py',
  'src/models/baseline_lstm.py': 'src/models/baseline_lstm.py',
  'src/models/surge_jepa_v7e.py': 'src/models/surge_jepa_v7e.py',
  # ---- pipeline scripts (data acquisition + cleaning) ----
  'scripts/download_gesla.py': 'scripts/pipeline/download_gesla.py',
  'scripts/fetch_era5.py': 'scripts/pipeline/fetch_era5.py',
  'scripts/batch_detide.py': 'scripts/pipeline/batch_detide.py',
  'scripts/build_static_attributes.py': 'scripts/pipeline/build_static_attributes.py',
  'scripts/scan_stations.py': 'scripts/pipeline/scan_stations.py',
  'scripts/make_audit_splits.py': 'scripts/pipeline/make_audit_splits.py',
  # ---- training + canonical evaluation ----
  'scripts/train_baseline_lstmq.py': 'scripts/train.py',
  'scripts/train_v7e.py': 'scripts/experiments/train_v7e.py',
  'scripts/eval_full.py': 'scripts/eval_full.py',
  'scripts/verify_numbers.py': 'scripts/verify_numbers.py',
  'scripts/verify_numbers_cont.py': 'scripts/verify_numbers_cont.py',
  'scripts/summarize_eval_logs.py': 'scripts/summarize_eval_logs.py',
  'scripts/audit2_numbers.py': 'scripts/audit2_numbers.py',
  'scripts/compare_cont.py': 'scripts/compare_cont.py',
  'scripts/bootstrap_cis.py': 'scripts/bootstrap_cis.py',
  # ---- experiment scripts ----
  'scripts/eval_forcing_only.py': 'scripts/experiments/eval_forcing_only.py',
  'scripts/fit_sigma_hat.py': 'scripts/experiments/fit_sigma_hat.py',
  'scripts/extract_eot20_statics.py': 'scripts/experiments/extract_eot20_statics.py',
  'scripts/fetch_gtsm.py': 'scripts/experiments/fetch_gtsm.py',
  'scripts/eval_gtsm.py': 'scripts/experiments/eval_gtsm.py',
  'scripts/eval_gtsm_symmetric.py': 'scripts/experiments/eval_gtsm_symmetric.py',
  'scripts/fetch_gefs_cycles.py': 'scripts/experiments/fetch_gefs_cycles.py',
  'scripts/eval_gefs.py': 'scripts/experiments/eval_gefs.py',
  'scripts/exp_trailing_baseline.py': 'scripts/experiments/exp_trailing_baseline.py',
  'scripts/eval_trailing.py': 'scripts/experiments/eval_trailing.py',
  'scripts/eval_pinball.py': 'scripts/experiments/eval_pinball.py',
  'scripts/eval_enso.py': 'scripts/experiments/eval_enso.py',
  'scripts/baseline_chronos.py': 'scripts/experiments/baseline_chronos.py',
  # ---- figures ----
  'scripts/pubstyle.py': 'scripts/figures/pubstyle.py',
  'scripts/plot_fig1_design.py': 'scripts/figures/plot_fig1_design.py',
  'scripts/plot_perlead_full.py': 'scripts/figures/plot_perlead_full.py',
  'scripts/plot_scaling_v2.py': 'scripts/figures/plot_scaling.py',
  'scripts/plot_case_studies.py': 'scripts/figures/plot_case_studies.py',
  'scripts/plot_random_windows.py': 'scripts/figures/plot_random_windows.py',
  'scripts/plot_figS1_zoom.py': 'scripts/figures/plot_figS1_zoom.py',
  'scripts/plot_gis_pack1.py': 'scripts/figures/plot_gis_pack1.py',
  'scripts/plot_gis_pack2.py': 'scripts/figures/plot_gis_pack2.py',
  'scripts/plot_gis_pack3.py': 'scripts/figures/plot_gis_pack3.py',
  'scripts/plot_gis_gtsm_sym.py': 'scripts/figures/plot_gis_gtsm_sym.py',
  # ---- audit documentation ----
  'docs/experiment_ledger.md': 'docs/experiment_ledger.md',
  'docs/protocol_symmetry_audit.md': 'docs/protocol_symmetry_audit.md',
  'docs/gtsm_symmetry_audit.md': 'docs/gtsm_symmetry_audit.md',
  'docs/audit2_results.md': 'docs/audit2_results.md',
  # ---- frozen splits + catalogs (pre-registration artifacts) ----
  'catalog/exp_split_final.csv': 'catalog/exp_split_final.csv',
  'catalog/exp_split_jp_v2.csv': 'catalog/exp_split_jp_v2.csv',
  'catalog/exp_split_eu_v2.csv': 'catalog/exp_split_eu_v2.csv',
  'catalog/exp_split_na_v2.csv': 'catalog/exp_split_na_v2.csv',
  'catalog/exp_split_oc_v2.csv': 'catalog/exp_split_oc_v2.csv',
  'catalog/exp_split_eu299v.csv': 'catalog/exp_split_eu299v.csv',
  'catalog/exp_split_n298v.csv': 'catalog/exp_split_n298v.csv',
  'catalog/exp_split_c298strat.csv': 'catalog/exp_split_c298strat.csv',
  'catalog/exp_split_c298usjp.csv': 'catalog/exp_split_c298usjp.csv',
  'catalog/exp_split_g64.csv': 'catalog/exp_split_g64.csv',
  'catalog/exp_split_g128.csv': 'catalog/exp_split_g128.csv',
  'catalog/exp_split_g256.csv': 'catalog/exp_split_g256.csv',
  'catalog/exp_split_g384.csv': 'catalog/exp_split_g384.csv',
  'catalog/exp_split_g512.csv': 'catalog/exp_split_g512.csv',
  'catalog/exp_split_g640.csv': 'catalog/exp_split_g640.csv',
  'catalog/exp_split_g640marine.csv': 'catalog/exp_split_g640marine.csv',
  'catalog/exp_split_marine_abl.csv': 'catalog/exp_split_marine_abl.csv',
  'catalog/static_attributes.csv': 'catalog/static_attributes.csv',
  'catalog/clean_stations.csv': 'catalog/clean_stations.csv',
  'catalog/processing_qc.csv': 'catalog/processing_qc.csv',
  'catalog/earthquakes_m7.csv': 'catalog/earthquakes_m7.csv',
  # ---- checkpoints (models of record) ----
  'data/raw/gtsm/gtsm_mapping.csv': 'outputs/gtsm_mapping.csv',
  'outputs/baseline_lstmq_v2final_best.pt': 'models/deploy_760_best.pt',
  'outputs/baseline_lstmq_v2fonly_best.pt': 'models/gaugefree_760_best.pt',
  'outputs/baseline_lstmq_v2rotjp_best.pt': 'models/rot_japan_best.pt',
  'outputs/baseline_lstmq_v2roteu_best.pt': 'models/rot_europe_best.pt',
  'outputs/baseline_lstmq_v2rotna_best.pt': 'models/rot_namerica_best.pt',
  'outputs/baseline_lstmq_v2rotoc_best.pt': 'models/rot_oceania_run1_best.pt',
  'outputs/baseline_lstmq_v2rotocr2_best.pt': 'models/rot_oceania_median_best.pt',
  'outputs/baseline_lstmq_v2rotocr3_best.pt': 'models/rot_oceania_run3_best.pt',
  'outputs/baseline_lstmq_v2eu299_best.pt': 'models/europe_intervention_299_best.pt',
  'outputs/baseline_lstmq_v2c298strat_best.pt': 'models/composition_298_49countries_best.pt',
  'outputs/baseline_lstmq_v2c298usjp_best.pt': 'models/composition_298_usa_japan_best.pt',
  'outputs/baseline_lstmq_v2g64ms_best.pt': 'models/ladder_64_matched_steps_best.pt',
  'outputs/baseline_lstmq_v2g256ms_best.pt': 'models/ladder_256_matched_steps_best.pt',
  'outputs/baseline_lstmq_v2n298_best.pt': 'models/factor_lstm_298_japan_heldout_best.pt',
  'outputs/surge_v7e_n298v_best.pt': 'models/factor_attention_v7e_298_japan_heldout_best.pt',
}

# evaluation artifacts of record (everything verify_numbers*.py reads) + figures
ARTIFACTS = [
  'eval_full_lstmq_v2final.csv', 'eval_full_lstmq_v2final.log',
  'eval_full_lstmq_v2final_xdom.csv', 'eval_full_lstmq_v2final_xdom.log',
  'eval_full_lstmq_v2g64.csv', 'eval_full_lstmq_v2g128.csv', 'eval_full_lstmq_v2g256.csv',
  'eval_full_lstmq_v2g384.csv', 'eval_full_lstmq_v2g512.csv', 'eval_full_lstmq_v2g512r2.csv',
  'eval_full_lstmq_v2g640.csv', 'eval_full_lstmq_v2g640marine.csv', 'eval_full_lstmq_v2marine.csv',
  'eval_full_lstmq_v2n298.csv',
  'eval_full_lstmq_v2rotjp.csv', 'eval_full_lstmq_v2rotjp.log',
  'eval_full_lstmq_v2roteu.csv', 'eval_full_lstmq_v2roteu.log',
  'eval_full_lstmq_v2rotna.csv', 'eval_full_lstmq_v2rotna.log',
  'eval_full_lstmq_v2rotoc.csv', 'eval_full_lstmq_v2rotocr2.csv', 'eval_full_lstmq_v2rotocr2.log',
  'eval_full_lstmq_v2rotocr3.csv',
  'eval_forcing_only_v2fonly.csv', 'eval_forcing_only_v2fonly_predscale.csv', 'eval_forcing_only_v2fonly_eot2.csv',
  'sigma_hat_test.csv', 'sigma_hat_eot_test.csv', 'eot20_statics_test.csv',
  'eval_gtsm.csv', 'eval_gtsm.log',
  'gfs_events.csv',   # eval_gefs.{csv,log} of the July preprint stay as committed in the release (anchor bug; corrected run = eval_gefs_v2)
  'eval_trailing_centered.csv', 'eval_trailing_trailing.csv', 'eval_trailing.log', 'trailing_qc.csv',
  'eval_pinball.csv', 'eval_pinball.log',
  'eval_enso_ocrot.csv', 'eval_enso_deploy.csv', 'eval_enso.log',
  'perlead_full.csv', 'seismic_exceedance_audit.csv',
  # figures of record (July preprint)
  'fig1a_map.pdf', 'fig1b_architecture.pdf', 'fig1c_example.pdf', 'perlead_full.pdf',
  'scaling_figure.pdf', 'case_studies_extremes.pdf', 'figS1_zoom.pdf', 'figS2_random.pdf',
  'gis_skillmap.pdf', 'gis_rotations.pdf', 'gis_density.pdf', 'gis_ppbay.pdf',
  'gis_gtsm_delta.pdf', 'gis_case_maps.pdf',
]
# ---- September 2026 revision: every headline checkpoint on all windows and on continuous windows only ----
_TAGS = ['lstmq_v2final', 'lstmq_v2rotjp', 'lstmq_v2roteu', 'lstmq_v2rotna', 'lstmq_v2rotoc', 'lstmq_v2rotocr2', 'lstmq_v2rotocr3',
         'lstmq_v2eu299', 'lstmq_v2g64', 'lstmq_v2g128', 'lstmq_v2g256', 'lstmq_v2g384', 'lstmq_v2g512', 'lstmq_v2g512r2',
         'lstmq_v2g640', 'lstmq_v2g640marine', 'lstmq_v2marine', 'lstmq_v2g64ms', 'lstmq_v2g256ms',
         'lstmq_v2c298strat', 'lstmq_v2c298usjp', 'lstmq_v2n298', 'v7e_n298v']
for _t in _TAGS:
    for _suf in ('', '_cont'):
        ARTIFACTS += [f'eval_full_{_t}{_suf}.csv', f'eval_full_{_t}{_suf}.log']
ARTIFACTS += [
  'eval_gtsm_symmetric.csv', 'eval_gtsm_symmetric.log', 'eval_gtsm_symmetric_cont.csv', 'eval_gtsm_symmetric_cont.log',
  'eval_forcing_only_v2fonly_cont.csv', 'eval_forcing_only_v2fonly_predscale_cont.csv', 'eval_forcing_only_v2fonly_eot2_cont.csv',
  'eval_forcing_only_v2final.csv', 'eval_forcing_only_v2final_cont.csv',
  'eval_chronos_bolt-small_masked.csv', 'eval_chronos_bolt-small_masked.log',
  'eval_chronos_bolt-small_masked_cont.csv', 'eval_chronos_bolt-small_masked_cont.log',
  'eval_pinball_cont.csv', 'eval_pinball_cont.log',
  'eval_gefs_v2.csv', 'eval_gefs_v2.log', 'gfs_inits_inwindow.csv',
  'perlead_full_cont.csv', 'compare_cont.txt', 'audit2_numbers.txt', 'eval_log_summary.txt',
  'bootstrap_cis.txt', 'bootstrap_cis_cont.txt',
  # figures of record (Nature Water draft, continuous windows)
  'gis_skillmap_cont.pdf', 'perlead_full_cont.pdf', 'gis_gtsm_delta_sym.pdf', 'gis_gtsm_delta_sym_cont.pdf',
  'gis_rotations_cont.pdf', 'scaling_figure_cont.pdf', 'gis_ppbay_cont.pdf',
]
ARTIFACTS = list(dict.fromkeys(ARTIFACTS))

if os.path.exists(DST) and not os.path.exists(f'{DST}/.git'): shutil.rmtree(DST)
missing = []
for s, d in MANIFEST.items():
    sp, dp = f'{SRC}/{s}', f'{DST}/{d}'
    if not os.path.exists(sp): missing.append(s); continue
    os.makedirs(os.path.dirname(dp), exist_ok=True); shutil.copy2(sp, dp)
for a in ARTIFACTS:
    sp, dp = f'{SRC}/outputs/{a}', f'{DST}/outputs/{a}'
    if not os.path.exists(sp): missing.append(f'outputs/{a}'); continue
    os.makedirs(os.path.dirname(dp), exist_ok=True); shutil.copy2(sp, dp)
print(f'staged {len(MANIFEST)+len(ARTIFACTS)-len(missing)} files -> {DST}')
if missing: print('MISSING:', missing)
