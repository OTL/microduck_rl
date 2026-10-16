# QDD 二足（Mjlab-Velocity-Flat-QddBiped）

自作の小型 QDD 二足（6.48 kg、立ち姿勢で高さ約 0.52 m、関節 10 個）に、平地で指令速度に追従して歩く方策を学習させる課題。microduck の歩行課題（`microduck_velocity_env_cfg.py`）のやり方をこのロボットに移したもの。microduck の 61 次元の方策ファミリーとは別物で、観測の並びも違う。

ロボット側の正本はロボットのリポジトリの `robot/model.py`（寸法・質量・関節範囲・モータ仕様・`knee_max`・`crouch_pose`）。

## ファイル

| ファイル | 内容 |
|---|---|
| `src/mjlab_microduck/robot/qdd_biped/qdd_biped.xml` | 学習用 MJCF（生成物。手で直さない） |
| `src/mjlab_microduck/robot/qdd_biped/generate_mjcf.py` | ロボットのリポジトリから上の XML を作る |
| `src/mjlab_microduck/robot/qdd_biped_constants.py` | ロボット設定（アクチュエータ、ゲイン、初期姿勢、接触） |
| `src/mjlab_microduck/tasks/qdd_biped_velocity_env_cfg.py` | 課題の設定と PPO の設定 |
| `src/mjlab_microduck/tasks/mdp.py`（末尾） | `knee_max` の torch 版、膝制限のコストと終了条件、歩容クロック（`GaitClockCommand`）と歩容の報酬 |
| `src/mjlab_microduck/tasks/symmetry.py`（末尾） | 左右反転の対応表（`qdd_biped_symmetry`、41 次元） |
| `src/mjlab_microduck/robot/qdd_biped/policy_io.json` | 実機コントローラ向けの入出力仕様（生成物。下の「方策の入出力仕様」） |
| `tests/test_qdd_biped_cfg.py` | 設定の検査（CPU で動く） |

技（お辞儀などのキーフレーム追従、`Mjlab-Skill-QddBiped`）は `docs/qdd_skill.md`。

XML の作り直し（`robot/model.py` を変えたとき）:

```
uv run python src/mjlab_microduck/robot/qdd_biped/generate_mjcf.py --robot-repo <ロボットのリポジトリ>
```

## モデル

- 関節の並び（行動・観測とも）: `l_hip_yaw, l_hip_roll, l_hip_pitch, l_knee, l_ankle_pitch, r_...`（同じ順）。
- 接触: 足裏の箱（150×70 mm、足首の真下）だけが床と摩擦付きで当たる。胴体は箱 2 つ、腿・すねはカプセル、膝モータは球で、脚どうしの自己衝突を検出して罰する。足裏以外が床に触れたら転倒として終了。
- 見た目はプリミティブ（円柱・箱）だけ。メッシュは使わない。
- 初期姿勢は `crouch_pose(45°)`（股 -22.5°、膝 45°、足首 -22.5°）、胴体原点の高さ 0.3542 m で足裏が床に着く（10 mm の着地パッド込み。2026-10-16 以前は 0.3442 m）。
- IMU は胴体の `imu` サイト（原点の上 0.1 m）。

## アクチュエータとゲイン

ドライバが MIT モードで PD を回す前提なので、MuJoCo 組み込みの位置 PD（物理ステップごと）を使い、出力はピークトルクで頭打ちにする。実機のドライバにも同じ Kp/Kd を設定すること（ONNX のメタデータ `joint_stiffness` / `joint_damping` に入っている）。

| 関節 | モータ | Kp [Nm/rad] | Kd [Nm·s/rad] | トルク上限 [Nm] | アーマチュア [kg·m²] |
|---|---|---|---|---|---|
| hip_yaw, hip_roll | GIM6010-8 | 30 | 1.0 | 11 | 3.84e-3 |
| hip_pitch, knee | GIM6010-8 | 40 | 1.2 | 11 | 3.84e-3 |
| ankle_pitch | GIM3510-8 | 20 | 0.6 | 6.28 | 1.5e-3 |

- 行動は「既定姿勢からのずれ × 0.25 rad」。フィルタは掛けない（microduck と同じ）。
- 指令の遅れは物理ステップ 0〜2 回分（0〜10 ms）。
- 確かめたこと: 素の MuJoCo で既定姿勢に PD をかけて 5 秒立たせると、足裏が平らなまま落ち着く。PD のたわみで胴体が後ろに約 4.5° 傾くが、これは方策が直す。

## 観測（actor 41 次元）

実機で取れるものだけを入れる。

| 項目 | 次元 | 備考 |
|---|---|---|
| base_ang_vel | 3 | IMU ジャイロ。取付誤差（環境ごとに最大 3°）、0〜1 ステップの遅れ、ノイズ ±0.05 |
| projected_gravity | 3 | IMU の姿勢から。取付誤差と遅れは上と同じ、ノイズ ±0.02 |
| joint_pos | 10 | 既定姿勢との差。エンコーダの零点ずれ ±0.01 rad、ノイズ ±0.005 |
| joint_vel | 10 | ノイズ ±0.3 |
| actions | 10 | 前回の行動 |
| command | 3 | vx, vy, wz |
| gait_phase | 2 | 歩容クロックの [sin 2πφ, cos 2πφ]。ノイズ・遅れなし（実機が自分で計算する） |

critic には、これに加えて胴体の並進速度、足の高さ・接地・接地力・滞空時間を入れる（学習のときだけ使う情報）。観測の正規化は ON で、`scripts/export.py` が ONNX に焼き込む。

## 指令

- vx −0.3〜0.5 m/s、vy −0.2〜0.2 m/s、wz −1〜1 rad/s。
- 15% の環境は「その場で旋回」（並進 0、|wz| は最大値の 0.4〜1 倍）。
- 30% の環境は「まっすぐ前進」（vx 0.3〜0.5 m/s、vy = wz = 0。mjlab の rel_forward_envs、テンプレートの既定は 20%）。
- 静止（指令ゼロ）の環境は 5% から始めて、750 反復で 20% まで増やす（1000 反復の学習に合わせて縮めた）。

## 歩容（人らしく歩かせる仕組み）

前回（3000 反復、報酬 ~123 で頭打ち）は歩けたが、足運びが人らしくなかった。そこで歩容クロックを入れた。

- **クロック**: 位相 φ ∈ [0, 1)。周期 **0.64 s**（50 Hz で 32 ステップ）。脚長 0.26 m に人の歩行周期をフルード数で縮めると約 0.6 s。乱数化はしない（実機と完全に同じにするため）。
- **進め方**: 毎ステップ、|v_xy| + |w_z| > 0.05 なら φ ← (φ + 0.02/0.64) mod 1、そうでなければ止める（静止中は両足接地）。リセット時の φ は乱数（どちらの足からでも踏み出せるように）。実機は 0 から始めてよい。
- **接地の予定**: 左脚の位相 φ、右脚は φ + 0.5。各脚は位相 [0, 0.4) で遊脚、残りが立脚。片脚支持が交互に来て、その間に 64 ms の両脚支持が 2 回入る。
- **遊脚の高さ**: 足裏の高さの目標は 5 cm × sin(π × 遊脚の進み具合)、立脚中は 0。
- **左右対称**: PPO に mirror loss（係数 0.5）を入れた。左右の脚は関節軸が同じなので、反転は「左右の入れ替え＋ hip_yaw / hip_roll の符号反転」。クロックは φ → φ + 0.5（sin, cos の符号反転）。対応表は MuJoCo で足先位置が鏡像になることをテストで確かめている。
- 歩幅は速度に応じて変わる（周期は固定）。0.5 m/s で一歩 0.16 m。

## 報酬

| 項目 | 重み | 内容 |
|---|---|---|
| track_linear_velocity | +2.0 | 並進速度の追従（std 0.25 m/s、粗い方） |
| track_linear_velocity_fine | +1.5 | 水平の並進速度だけの追従（std 0.1 m/s、細かい方。上下の揺れは数えない） |
| track_angular_velocity | +2.0 | 旋回速度の追従（std 0.5 rad/s） |
| upright | +2.0 | 胴体が立っていること（std² 0.05） |
| pose | +1.0 | 既定姿勢に近いこと。歩くときは緩める。足首ロールが無いので、横の重心移動に使う hip_roll は歩行時に 0.15 まで許す |
| **gait_contact** | +1.5 | 各足の接地がクロックの予定と合っている割合（遊脚 = 浮いている、立脚・静止 = 着いている） |
| **gait_swing_height** | +1.0 | 足裏の高さが目標（遊脚 5 cm の正弦、立脚 0）に近いこと（std 3 cm） |
| foot_swing_height | −0.25 | 着地時の最高到達高さが 5 cm からずれた分 |
| foot_slip | −1.0 | 接地中の足の滑り速度²（静止中も） |
| **feet_flat** | −2.0 | 接地中または高さ 2 cm 未満の足裏の傾き（平らに着地させる） |
| **touchdown_velocity** | −2.0 | 着地直前の足の下向き速度² |
| soft_landing | −1e-5 | 着地の衝撃力（mjlab そのまま、静止中も） |
| **feet_spacing** | −10 | 左右の足裏の横間隔（進行方向基準）が 8〜20 cm の外に出た分 [m]（交差・大股開きを防ぐ） |
| **hip_yaw_roll** | −1.0 | hip_yaw / hip_roll の既定値からのずれ² |
| action_rate_l2 | −0.05 → −0.3 | 行動の変化。250/500/750 反復で −0.1/−0.2/−0.3 |
| joint_torques | −2e-4 | トルクの二乗和 |
| body_ang_vel / angular_momentum | −0.05 / −0.02 | 胴体の揺れ |
| dof_pos_limits | −1.0 | 関節範囲の端 |
| self_collisions | −1.0 | 脚どうしの衝突（10 N 超） |
| knee_limit | −10 | 膝が `knee_max(hip_pitch)` − 0.05 rad を超えた分 [rad] |

太字が歩容クロックで足したもの。

2026-10-15 の見直し（gait1 の結果から）:

- 速度追従: gait1 を学習と同じシミュレータ（play の公称条件）で動かすと、指令 0.3 / 0.5 m/s に対して 0.284 / 0.441 m/s。粗いカーネル（std 0.25）だけだと 0.5 指令で 0.44 でも報酬の 70% が出るので、細かいカーネル（std 0.1、水平成分だけ）を足した。なお sim2sim の「平均 vx 0.18」は、旋回区間と停止区間を含めた世界座標の dx ÷ 時間で、直進区間の胴体座標の速度は 0.29 m/s だった（追従そのものは悪くなかった）。
- 遊脚の高さ: gait1 は遊脚の高さが平均 1.3 cm（目標 5 cm）。std 2 cm のガウスでは 3 cm 以上離れると勾配がほぼ無いので 3 cm に広げた。
テンプレートの air_time（+2.0）と foot_clearance は外した（どんな踏み方でも報酬が出てしまうため、クロックの項に置き換え）。

`knee_max` は `robot/model.py` の段階関数そのまま（股ピッチ > 0° なら膝 80° まで、> −15° で 90°、> −30° で 100°、> −45° で 110°、それ以下で 120°）。

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

play（書き出し・yatai の動画とスナップショット）は公称のロボットで動かす: 上の乱数化（摩擦・重心・質量・Kp/Kd・トルク上限・アーマチュア・関節摩擦・押し）を全部外し、観測のノイズ・遅れ・IMU 取付誤差・エンコーダ零点ずれも外し、指令の遅れは 5 ms 固定。エピソードは打ち切らない。指令は乱数ではなく決まった予定（`PLAY_COMMAND_SCHEDULE`、15 s で繰り返し）: 0〜1 s 静止 → 1〜5 s 前進 0.3 m/s → 5〜9 s 前進 0.5 m/s → 9〜12 s その場で左旋回 0.8 rad/s → 12〜15 s 停止。動画は `--video-s 15` で 1 周分になる。観測の並びは変わらない（policy_io.json も同じ）。

## 方策の入出力仕様（実機の Rust コントローラ向け）

`src/mjlab_microduck/robot/qdd_biped/policy_io.json` が正本。`build_policy_io()`（`qdd_biped_velocity_env_cfg.py`）から生成し、テストで「ファイル = 生成結果」「観測の並びと次元 = 実際の環境の actor 入力」「関節順・既定姿勢・Kp/Kd = ONNX メタデータ」を確かめている。観測・行動・ゲインを変えたら作り直す:

```
python -m uv run python -m mjlab_microduck.tasks.qdd_biped_velocity_env_cfg
```

要点:

- 制御周期 50 Hz。ONNX の入力 `obs` [1, 41]（正規化は ONNX 内。生の値を入れる）、出力 `actions` [1, 10]（平均）。
- 目標角 = default_joint_pos + 0.25 × action。クリップ・フィルタなし。関節順は上の「モデル」のとおり。
- 観測（この順、履歴なし、スケーリングなし）: base_ang_vel 3（胴体座標の角速度 rad/s）、projected_gravity 3（胴体座標の重力単位ベクトル、直立で (0,0,−1)）、joint_pos 10（q − 既定値）、joint_vel 10（rad/s）、actions 10（前回の生の出力）、command 3（vx, vy, wz）、gait_phase 2（sin, cos）。
- クロックの更新規則・周期・閾値、指令範囲、Kp/Kd/トルク上限、関節範囲も JSON に入っている。
- ONNX のメタデータに JSON そのもの（`policy_io`）と SHA-256（`policy_io_sha256`）を入れる（`scripts/export.py` と、学習中にチェックポイントごとに出す ONNX の両方）。

## ローカルで確かめたこと（GPU なしの Windows、CPU）

- `tests/test_qdd_biped_cfg.py`（15 件）: 報酬の符号、観測の並び、反転表（往復で元に戻る・MuJoCo の足先が鏡像）、クロックが歩行中だけ 0.02/0.64 ずつ進み静止中は止まる、静止姿勢で歩容の報酬が満点、policy_io.json と実環境の一致。
- `CUDA_VISIBLE_DEVICES= python -m uv run train Mjlab-Velocity-Flat-QddBiped --env.scene.num-envs 8 --agent.max_iterations 3 --agent.logger tensorboard` が CPU で最後まで動く（symmetry loss も出る）。
- `scripts/export.py ... --checkpoint-file ... --num-envs 1 --onnx-file ...` で ONNX（入力 41、出力 10、`policy_io` メタデータ付き）が出て、onnxruntime で推論できる。

学習が実際に人らしく歩くかは GPU で回すまでわからない。

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

本番の目安: `--envs 4096 --iters 1000`。カリキュラム（行動変化の罰、静止環境の割合）は 750 反復で最終値になるよう縮めてある。`max_iterations` の既定も 1000。
