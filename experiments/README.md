# Experiments for the publication

Scripts that produce every number the manuscript still leaves open
(*A Multi-Camera Edge-AI Perception–Actuation System for Real-Time Selective
Vinasse Application in Sugarcane Fields*). Each script prints its results and
writes CSV/JSON under `experiments/results/` (git-ignored: results stay
private until publication). `e10_paper_macros.py` turns those files into LaTeX
macros for the placeholders of the abstract and introduction.

| Exp. | Script | Paper claim / placeholder | Runs on | Needs |
|---|---|---|---|---|
| E0 | `e0_audit_dataset.py`, `e0_label_agreement.py` | dataset description, leakage-free split, agreement (κ) between labelling variants | CPU | Roboflow folder datasets |
| E1 | `e1_gt_sample.py`, `e1_annotate.py`, `e1_agreement.py` | independent GT: *n images, two annotators, Cohen's κ* | CPU + 2 people | raw frames or field snapshots |
| E2 | `e2_eval_gt.py` | *accuracy [XX %]*, *exceeded the colour index that supervised it by [XX] pp* | CPU/GPU | GT + trained weights |
| E3 | `e3_ablation_curation.py` | *ablation over the curated fraction [0/10/50/100 %]* | GPU (RTX 3070) | pseudo-labelled **and** curated labelling of the same frames (matched by frame name, any split) |
| E4 | `e4_roboflow_download.py`, `e4_train_compare.py` | *cascaded HSV, LAB and ExGR masks replace most manual labelling* (one model per Roboflow labelling variant) | GPU | the 10 Roboflow projects |
| E5 | `e4_train_compare.py --arch …` | comparison of the deployed YOLO26n-cls with YOLO26s-cls / MobileNetV3 / EfficientNet-B0 / ResNet18 | GPU | one dataset |
| E6 | `e6_benchmark_latency.py` | *throughput [XX] images per second*, inference latency, energy per image | Jetson Orin Nano | `.engine` files |
| E7 | `e7_e2e_latency.py` | *camera-to-valve latency [XX ms]*, inter-camera skew, camera lead distance | Jetson + PLC | `decision_logger.py` logs, Modbus host |
| E8 | `e8_savings.py` | *estimated [XX %] reduction in applied volume*, debounce trade-off | CPU | field logs or a contiguous recording |
| E9 | `e9_field_ab.py` | YOLO26 vs. the deployed index/NDVI pipeline on the same field frames, day vs. night | CPU | field logs of both pipelines + labelled snapshots |
| E10 | `e10_paper_macros.py` | fills the `\todoNote{XX}` placeholders | CPU | outputs of E1–E9 |

`mexp/` holds the shared code: data listing and temporal block split
(`data.py`), metrics with Wilson / bootstrap CIs, McNemar and Cohen's κ
(`metrics.py`), the colour labellers imported from `labelbox_utils`
(`pseudolabel.py`), uniform batched inference for `.pt/.engine/.onnx/.pth`
(`models.py`), YOLO26 training with the 320×96 rectangular protocol
(`train.py`), torchvision baselines (`tv_models.py`) and the valve policies
(`debounce.py`). Training hyper-parameters are shared through
`configs/train.yaml` and reproduce the deployed model
(`runs/classify/magdalena/plant_cls_v4/args.yaml`).

## Protocol decisions

* **Temporal leakage.** The Roboflow splits are random per frame, so most test frames have a
  training frame within ±2 frames (E0 reports the fraction). Every
  training run for the paper uses the block split (`--block-split`, blocks of
  300 frames = 20 s, 30-frame guard band). The assignment depends only on the
  frame number, so all Roboflow variants share the same held-out blocks.
* **Ground truth.** Labels in `dataset/` (HSV) and `dataset2/` (cascade + manual
  fixes) disagree on a share of the frames they have in common, so a test
  split labelled by one index favours models trained on that index. Report accuracy
  only on the independent GT of E1, sampled from held-out blocks
  (`--block-test`) or, better, from new field passes (day and night) recorded
  with `decision_logger.py`.
* **Evaluation at 320×96.** `model.val()` evaluates at a square size; all
  scripts here predict on 320×96 stretched inputs (as Roboflow built the data
  and as the C++ node feeds TensorRT).
* **FLOPs** are counted with `torch.utils.flop_counter` at 96×320
  (YOLO26n-cls: 0.24 GFLOPs, YOLO26s-cls: 0.90 GFLOPs; Ultralytics reports them at 640×640).

## Order of execution

```bash
# 0. Data (ask for ROBOFLOW_API_KEY in the environment)
python experiments/e4_roboflow_download.py --list <workspace>
python experiments/e4_roboflow_download.py --out roboflow <workspace>/<project>/<version> ...
python experiments/e0_audit_dataset.py --data dataset2 --make-block-split dataset2_block
python experiments/e0_label_agreement.py --dataset roboflow/* --reference sugarcane_exg_hsv_lab_v3-2-100

# 1. Independent GT (two annotators, blind)
python experiments/e1_gt_sample.py --source <raw_frames_or_snapshots> [--block-test] --n 300
python experiments/e1_annotate.py --manifest experiments/gt/manifest.csv --annotator A
python experiments/e1_annotate.py --manifest experiments/gt/manifest.csv --annotator B
python experiments/e1_agreement.py --manifest experiments/gt/manifest.csv \
    --a experiments/gt/labels_A.csv --b experiments/gt/labels_B.csv [--adjudication ...]

# 2-5. Training and evaluation (RTX 3070)
python experiments/e4_train_compare.py --gt experiments/gt/gt.csv --block-split \
    --dataset cascade_curated=dataset2 --seeds 0,1,2 --export-onnx
python experiments/e2_eval_gt.py --gt experiments/gt/gt.csv --model yolo26n_curated=<best.pt>
python experiments/e3_ablation_curation.py --pseudo roboflow/sugarcane_exg_hsv_lab_v3 \
    --curated roboflow/sugarcane_exg_hsv_lab_v3-2-100 \
    --gt experiments/gt/gt.csv --fractions 0,10,25,50,100 --seeds 0,1,2
python experiments/e4_train_compare.py --gt experiments/gt/gt.csv --block-split \
    --dataset hsv_v1=roboflow/sugarcanehsv_v1 --dataset exg_v1=roboflow/sugarcaneexg_v1 ...
python experiments/e4_train_compare.py --gt experiments/gt/gt.csv --dataset curated=dataset2_block \
    --arch yolo26n-cls.pt --arch yolo26s-cls.pt --arch mobilenet_v3_small --arch efficientnet_b0 --arch resnet18 --export-onnx

# 6. Jetson (sudo jetson_clocks; build every engine with trtexec, batch 1..5)
python experiments/e6_benchmark_latency.py --engine best_fp16.engine --tag yolo26n_fp16

# 7-9. Field passes (see "Field protocol")
python experiments/e7_e2e_latency.py --log logs/decisions_*.csv --modbus-host 192.168.11.60 \
    --valve-open-ms <measured> --valve-close-ms <measured> --speed-kmh <v>
python experiments/e8_savings.py --log logs/decisions_*.csv --speed-kmh <v> --flow-lpm <q> \
    --gt-metrics experiments/results/e2_gt/metrics.json --model-name yolo26n_curated
python experiments/e9_field_ab.py --gt experiments/gt_field/gt.csv --log logs/decisions_*.csv \
    --camera-order camera_55,camera_56,camera_57,camera_58,camera_59

# 10. Numbers for the manuscript
python experiments/e10_paper_macros.py --model yolo26n_curated --out paper/results_macros.tex
```

## Field protocol

1. Launch Magdalena-Camera, `cameraNode` and `inference_node` with
   `ros2_ws/config/magdalena_yolo.yaml`, the Magdalena-CV-Pipeline graph
   (index/NDVI, publishing `/threshold<i>`), and **one** `modbus_controller`
   driving the valves. The other pipeline runs in shadow mode (logged only).
2. Run `ros2_ws/tools/decision_logger.py -p log_index:=true -p snapshot_dir:=...`
   on the Jetson. Record day and night passes, noting speed, field and
   implement lighting.
3. Valve actuation time: drive a coil from `e7_e2e_latency.py --modbus-host`
   and film valve + LED at ≥240 fps (or scope the solenoid current) to obtain
   `--valve-open-ms` / `--valve-close-ms`.
4. Label a sample of the snapshots with E1 (they never entered training) and
   run E8/E9.
