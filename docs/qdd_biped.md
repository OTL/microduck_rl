# QDD 二足（Mjlab-Velocity-Flat-QddBiped）

自作の小型 QDD 二足（5.66 kg、立ち姿勢で高さ約 0.52 m、関節 10 個）に、平地で指令速度に追従して歩く方策を学習させる課題。microduck の歩行課題（`microduck_velocity_env_cfg.py`）のやり方をこのロボットに移したもの。microduck の 61 次元の方策ファミリーとは別物で、観測の並びも違う。

ロボット側の正本はロボットのリポジトリの `robot/model.py`（寸法・質量・関節範囲・モータ仕様・`knee_max`・`crouch_pose`）。

## ファイル

| ファイル | 内容 |
|---|---|
| `src/mjlab_microduck/robot/qdd_biped/qdd_biped.xml` | 学習用 MJCF（生成物。手で直さない） |
| `src/mjlab_microduck/robot/qdd_biped/generate_mjcf.py` | ロボットのリポジトリから上の XML を作る |
| `src/mjlab_microduck/robot/qdd_biped_constants.py` | ロボット設定（アクチュエータ、ゲイン、初期姿勢、接触） |
| `src/mjlab_microduck/tasks/qdd_biped_velocity_env_cfg.py` | 課題の設定と PPO の設定 |
| `src/mjlab_microduck/tasks/mdp.py`（末尾） | `knee_max` の torch 版、膝制限のコストと終了条件 |
| `tests/test_qdd_biped_cfg.py` | 設定の検査（CPU で動く） |

XML の作り直し（`robot/model.py` を変えたとき）:

```
uv run python src/mjlab_microduck/robot/qdd_biped/generate_mjcf.py --robot-repo <ロボットのリポジトリ>
```

## モデル

- 関節の並び（行動・観測とも）: `l_hip_yaw, l_hip_roll, l_hip_pitch, l_knee, l_ankle_pitch, r_...`（同じ順）。
- 接触: 足裏の箱（150×70 mm、足首の真下）だけが床と摩擦付きで当たる。胴体は箱 2 つ、腿・すねはカプセル、膝モータは球で、脚どうしの自己衝突を検出して罰する。足裏以外が床に触れたら転倒として終了。
- 見た目はプリミティブ（円柱・箱）だけ。メッシュは使わない。
- 初期姿勢は `crouch_pose(45°)`（股 -22.5°、膝 45°、足首 -22.5°）、胴体原点の高さ 0.3442 m で足裏が床に着く。
- IMU は胴体の `imu` サイト（原点の上 0.1 m）。

## アクチュエータとゲイン

ドライバが MIT モードで PD を回す前提なので、MuJoCo 組み込みの位置 PD（物理ステップごと）を使い、出力はピークトルクで頭打ちにする。実機のドライバにも同じ Kp/Kd を設定すること（ONNX のメタデータ `joint_stiffness` / `joint_damping` に入っている）。

| 関節 | モータ | Kp [Nm/rad] | Kd [Nm·s/rad] | トルク上限 [Nm] | アーマチュア [kg·m²] |
|---|---|---|---|---|---|
| hip_yaw, hip_roll | GIM6010-8 | 30 | 1.0 | 11 | 3.84e-3 |
| hip_pitch, knee | GIM6010-8 | 40 | 1.2 | 11 | 3.84e-3 |
| ankle_pitch | GIM4310-10 | 20 | 0.6 | 5.6 | 1.5e-3 |

- 行動は「既定姿勢からのずれ × 0.25 rad」。フィルタは掛けない（microduck と同じ）。
- 指令の遅れは物理ステップ 0〜2 回分（0〜10 ms）。
- 確かめたこと: 素の MuJoCo で既定姿勢に PD をかけて 5 秒立たせると、足裏が平らなまま落ち着く。PD のたわみで胴体が後ろに約 4.5° 傾くが、これは方策が直す。

## 観測（actor 39 次元）

実機で取れるものだけを入れる。

| 項目 | 次元 | 備考 |
|---|---|---|
| base_ang_vel | 3 | IMU ジャイロ。取付誤差（環境ごとに最大 3°）、0〜1 ステップの遅れ、ノイズ ±0.05 |
| projected_gravity | 3 | IMU の姿勢から。取付誤差と遅れは上と同じ、ノイズ ±0.02 |
| joint_pos | 10 | 既定姿勢との差。エンコーダの零点ずれ ±0.01 rad、ノイズ ±0.005 |
| joint_vel | 10 | ノイズ ±0.3 |
| actions | 10 | 前回の行動 |
| command | 3 | vx, vy, wz |

critic には、これに加えて胴体の並進速度、足の高さ・接地・接地力・滞空時間を入れる（学習のときだけ使う情報）。観測の正規化は ON で、`scripts/export.py` が ONNX に焼き込む。

## 指令

- vx −0.3〜0.5 m/s、vy −0.2〜0.2 m/s、wz −1〜1 rad/s。
- 15% の環境は「その場で旋回」（並進 0、|wz| は最大値の 0.4〜1 倍）。
- 静止（指令ゼロ）の環境は 5% から始めて、1500 反復で 20% まで増やす。

## 報酬

| 項目 | 重み | 内容 |
|---|---|---|
| track_linear_velocity | +2.0 | 並進速度の追従（std 0.25 m/s） |
| track_angular_velocity | +2.0 | 旋回速度の追従（std 0.5 rad/s） |
| upright | +2.0 | 胴体が立っていること（std² 0.05） |
| pose | +1.0 | 既定姿勢に近いこと。歩くときは緩める。足首ロールが無いので、横の重心移動に使う hip_roll は歩行時に 0.15 まで許す |
| air_time | +2.0 | 片足の滞空 0.15〜0.45 s |
| foot_clearance / foot_swing_height | −2.0 / −0.25 | 振り足の高さ 4 cm |
| foot_slip | −0.2 | 接地中の足の滑り |
| action_rate_l2 | −0.05 → −0.3 | 行動の変化。1500 反復かけて強める |
| joint_torques | −2e-4 | トルクの二乗和 |
| body_ang_vel / angular_momentum | −0.05 / −0.02 | 胴体の揺れ |
| dof_pos_limits | −1.0 | 関節範囲の端 |
| self_collisions | −1.0 | 脚どうしの衝突（10 N 超） |
| knee_limit | −10 | 膝が `knee_max(hip_pitch)` − 0.05 rad を超えた分 [rad] |

`knee_max` は `robot/model.py` の段階関数そのまま（股ピッチ > 0° なら膝 85° まで、> −15° で 90°、> −30° で 100°、> −45° で 110°、それ以下で 120°）。

## 終了条件

時間切れ（20 s）、胴体の傾き 60° 超、胴体の高さ 0.20 m 未満、足裏以外が床に接触、膝が `knee_max` を 0.15 rad 超えた、物理状態が NaN。

## ドメインランダム化

| 対象 | 範囲 |
|---|---|
| 足裏の摩擦 | 0.4〜1.2（起動時） |
| 胴体の重心 | x ±2 cm、y ±1.5 cm、z ±2 cm（起動時） |
| 胴体の質量・慣性 | ×0.9〜1.1（起動時） |
| Kp / Kd | ×0.85〜1.15 / ×0.8〜1.2（リセットごと） |
| トルク上限 | ×0.8〜1.05（電池の電圧降下など） |
| アーマチュア | ×0.8〜1.2 |
| 関節の摩擦 | ×0.5〜2.0（XML では 0.05 Nm） |
| 指令の遅れ | 0〜10 ms |
| 押し | 3〜6 s ごとに水平速度 ±0.3 m/s |
| IMU 取付誤差、エンコーダ零点ずれ、観測ノイズ | 上の「観測」のとおり |

play（書き出し・動画）では押しと観測ノイズを外し、エピソードを打ち切らない。

## ローカルで確かめたこと（GPU なしの Windows、CPU）

- `uv sync`、課題の登録と設定の組み立て、`tests/test_qdd_biped_cfg.py`（8 件）。
- `uv run train Mjlab-Velocity-Flat-QddBiped --env.scene.num-envs 64 --agent.max_iterations 5 --agent.logger tensorboard --gpu-ids None` が CPU で最後まで動き、チェックポイントが出る。
- 行動ゼロ（既定姿勢を保持）で 3 秒、8 環境とも転倒・自己衝突なし。
- `scripts/export.py ... --checkpoint-file ... --num-envs 1 --onnx-file ...` で ONNX（入力 39、出力 10）が出る。yatai の `worker/duck_video.py` で動画も出る。

学習が実際に歩きを覚えるかは GPU で長く回すまでわからない。

## yatai での動かし方

スモークテスト（64 環境・5 反復、数十円程度）:

```
python3 bench_duck.py --task Mjlab-Velocity-Flat-QddBiped --envs 64 --iters 5
```

ただし、この課題は fork（https://github.com/OTL/microduck_rl）のブランチ `yatai/qdd-biped` にしか無い。yatai の GPU 側（`worker/duck_handler.py`）は入力の `ref` でブランチを取ってくるが、`bench_duck.py` には今のところ `ref` を渡す引数が無い。次のどちらかが要る。

1. `bench_duck.py` に `--ref` を足して `job_input["ref"] = a.ref` を送る（推奨）。そのうえで
   `python3 bench_duck.py --task Mjlab-Velocity-Flat-QddBiped --envs 64 --iters 5 --ref yatai/qdd-biped`
2. あるいは、このブランチを焼き込みの版（`/duck_ref`）にしたイメージを作り直す。

どちらの場合も、先に `yatai/qdd-biped` を fork に push しておく必要がある。`pyproject.toml` と `uv.lock` は変えていないので、ブランチに切り替えても依存の入れ直しは起きない。

本番の目安: `--envs 4096 --iters 3000`（microduck と同程度）から。
