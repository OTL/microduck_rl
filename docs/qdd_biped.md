# QDD 二足（Mjlab-Velocity-Flat-QddBiped）

自作の小型 QDD 二足（6.61 kg、立ち姿勢で高さ約 0.53 m、関節 10 個）に、平地で指令速度に追従して歩く方策を学習させる課題。microduck の歩行課題（`microduck_velocity_env_cfg.py`）のやり方をこのロボットに移したもの。microduck の 61 次元の方策ファミリーとは別物で、観測の並びも違う。

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
- 接触: 足裏の箱（150×70×20 mm、足首の真下。PA12 の足裏の下に 10 mm の TPU パッド付き、足首軸から接地面まで 50 mm）だけが床と摩擦付きで当たる。胴体は箱 2 つ、腿・すねはカプセル、膝モータは球で、脚どうしの自己衝突を検出して罰する。足裏以外が床に触れたら転倒として終了。
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
| track_linear_velocity_fine | +1.2 | 水平の並進速度だけの追従（std 0.1 m/s、細かい方。上下の揺れは数えない） |
| track_angular_velocity | +2.0 | 旋回速度の追従（std 0.5 rad/s） |
| **track_angular_velocity_fine** | +1.0 | ヨー角速度だけの追従（std 0.15 rad/s、細かい方） |
| upright | +2.0 | 胴体が立っていること（std² 0.05） |
| pose | +1.0 | 既定姿勢に近いこと。歩くときは緩める。足首ロールが無いので、横の重心移動に使う hip_roll は歩行時に 0.15 まで許す |
| **gait_contact** | +1.5 | 各足の接地がクロックの予定と合っている割合（遊脚 = 浮いている、立脚・静止 = 着いている）。止まってから 0.6 s は数えない（足をそろえる一歩を許す） |
| **gait_swing_height** | +2.0 | 足裏の高さが目標（遊脚 5 cm の正弦、立脚 0）に近いこと（std 3 cm） |
| **swing_clearance** | −10 | 遊脚の足裏が最低高さ 3.5 cm × sin(π × 遊脚の進み具合) より低い分 [m]（片側だけ。高い分はただ） |
| **stand_square** | +2.0 | 静止中（クロック停止中）だけ：左右の足裏の前後のずれ（std 2 cm）、左右の股ピッチの差（std 0.05 rad）、荷重の左右差（左の割合と 0.5 の差、std 0.15）、既定姿勢からのずれ（二乗和、std 0.2 rad）の 4 つのガウスの平均。前後のずれと股ピッチの差は、細いガウスと 3 倍幅のガウスの平均（gait4） |
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
2026-10-16 の見直し（gait2 の結果から。gait3）:

- gait2 は速度追従は良い（0.306 / 0.500 m/s）が、遊脚が低い（平均 0.6 cm、最大 2〜3 cm。すり足）。細かい速度追従（1.5）が遊脚の高さ（1.0）より稼げていたので、遊脚の高さを 2.0 に上げ、細かい速度追従を 1.2 に下げ、最低高さを下回った分の罰（swing_clearance）を足した。
- その場旋回が 0.70 rad/s（指令 0.8）。粗いカーネル（std 0.5）ではこれでも 96% 出るので、ヨーだけの細かいカーネル（std 0.15）を足した。
- 止まると足が前後にずれたまま（股ピッチの差 ~0.18 rad）で、そこから技（お辞儀）を始めるとヨーが ~7° 回った。静止中だけ「足をそろえる」報酬（stand_square）を足した。クロックは環境でも実機のコントローラでも指令が 0 になった瞬間にその位相で止まる（コントローラの規則と同じ。変えていない）ので、止まってから 0.6 s は gait_contact が両足接地を求めず、一歩でそろえ直せるようにした（`GaitClockCommand.stand_time`。観測には入れない）。

2026-10-17 の見直し（足裏パッドの追加。gait4）:

- ロボットのリポジトリ 31cceba で足裏の下に 10 mm の TPU パッド（95A）が付いた。XML を作り直した（足の箱の中心 z −0.03 → −0.04、足 0.09 → 0.139 kg、ヨー・ロールのブラケットが A7075、全体 6.607 kg、脚が 10 mm 長い）。`STAND_BASE_Z` は 0.3442 → 0.3542 m。
- 接触の柔らかさの乱数化（`foot_solref`）: 足の箱の solref の時定数をリセットごとに 4〜20 ms（左右共通）。パッドの柔らかさは MuJoCo の既定（20 ms）くらい。足の箱は priority 1 なので、足と床の接触にはこの値がそのまま使われる。ただし MuJoCo の refsafe で時定数は 2 × timestep 以上に切り上げられるので、学習（5 ms）では 4〜10 ms は 10 ms として効く（sim2sim のブリッジは 2 ms なので 4 ms まで効く）。
- 止まったときの足そろえ: gait3 は遊脚の途中で止まると、そろうまで最大 3.7 s かかった。stand_square の前後のずれ（std 2 cm）と股ピッチの差（std 0.05 rad）のガウスは、よくある止まり方（8 cm・0.18 rad ずれ）だと std の 4 倍離れていて勾配が無い。そこで 3 倍幅のガウスとの平均にした（`STAND_COARSE_SCALE`）。クロックの止め方（コントローラと同じ規則）は変えていない。

テンプレートの air_time（+2.0）と foot_clearance は外した（どんな踏み方でも報酬が出てしまうため、クロックの項に置き換え）。

`knee_max` は `robot/model.py` の段階関数そのまま（股ピッチ > 0° なら膝 80° まで、> −15° で 90°、> −30° で 100°、> −45° で 110°、それ以下で 120°）。

## 終了条件

時間切れ（20 s）、胴体の傾き 60° 超、胴体の高さ 0.20 m 未満、足裏以外が床に接触、膝が `knee_max` を 0.15 rad 超えた、物理状態が NaN。

## ドメインランダム化

| 対象 | 範囲 |
|---|---|
| 足裏の摩擦 | 0.4〜1.2（起動時） |
| 足裏の接触の時定数（solref） | 4〜20 ms（リセットごと、左右共通。学習の 5 ms ステップでは 10 ms 未満は 10 ms） |
| 胴体の重心 | x ±2 cm、y ±1.5 cm、z ±2 cm（起動時） |
| 胴体の質量・慣性 | ×0.9〜1.1（起動時） |
| Kp / Kd | ×0.85〜1.15 / ×0.8〜1.2（リセットごと） |
| トルク上限 | ×0.8〜1.05（電池の電圧降下など） |
| アーマチュア | ×0.8〜1.2 |
| 関節の摩擦 | ×0.5〜2.0（XML では 0.05 Nm） |
| 指令の遅れ | 0〜10 ms |
| 押し | 3〜6 s ごとに水平速度 ±0.3 m/s |
| IMU 取付誤差、エンコーダ零点ずれ、観測ノイズ | 上の「観測」のとおり |

play（書き出し・yatai の動画とスナップショット）は公称のロボットで動かす: 上の乱数化（摩擦・接触の時定数・重心・質量・Kp/Kd・トルク上限・アーマチュア・関節摩擦・押し）を全部外し、観測のノイズ・遅れ・IMU 取付誤差・エンコーダ零点ずれも外し、指令の遅れは 5 ms 固定。エピソードは打ち切らない。指令は乱数ではなく決まった予定（`PLAY_COMMAND_SCHEDULE`、15 s で繰り返し）: 0〜1 s 静止 → 1〜5 s 前進 0.3 m/s → 5〜9 s 前進 0.5 m/s → 9〜12 s その場で左旋回 0.8 rad/s → 12〜15 s 停止。動画は `--video-s 15` で 1 周分になる。観測の並びは変わらない（policy_io.json も同じ）。

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

- `tests/test_qdd_biped_cfg.py`（25 件）: 報酬の符号、
観測の並び、反転表（往復で元に戻る・MuJoCo の足先が鏡像）、クロックが歩行中だけ 0.02/0.64 ずつ進み静止中は止まる、静止姿勢で歩容の報酬が満点、policy_io.json と実環境の一致。
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

## 腕つき（2026-10-18）

腕で両足をそろえて起き上がれるように、肩と肘の 2 軸の腕（肩の軸から手先まで 25 cm＝上腕 10＋前腕 15、手先に TPU のパッド、モータは両方 GIM3510-8、腕 1 本 0.74 kg、全体 6.61 → 8.10 kg。10/18 OTLさん承認の CAD 第 2 版の値）を付けたモデルと課題。長さ 20 cm までは探索（`scripts/qdd_getup_search/`）で決め、25 cm は CAD で決めた。手本（getup_demo_traj_arms.npz）は、10/18 夜に 25 cm の腕（`gpu_cem.py --cad`、肘は前曲げ）で探し直した仰向けの起き上がり（20 cm の候補 B から --asym --sd0 0.04 で詰め直し、速さの制限ありの再生で 32 回中 27 回立つ。結果 /mnt/project-files/robo/getup/arms/arms25_supine_cad_from20_sd004.npz、動画 arms25_supine.gif）。肩を後ろに 2 rad まで回して手を床につき、肘はほとんど使わない。--cad の対称探索は仰向け・うつ伏せとも立てなかった。ロボットのリポジトリの `robot/model.py` にはまだ入っていない（`robot/qdd_biped_constants.py` の `add_arms`）。

| 課題 | 腕 | 観測 / 行動 |
|---|---|---|
| `Mjlab-Velocity-Flat-QddBipedArms` | 方策が動かす（腕の姿勢はゆるい Gaussian、左右反転の mirror loss も腕つき） | 53 / 14（腕なしの policy_io に腕を足したものを ONNX に入れる） |
| `Mjlab-Velocity-Rough-QddBipedArms` | 同上 | 53 / 14 |
| `Mjlab-Skill-Hop-QddBipedArms` | 方策が動かす（脚は表の動き＋補正、腕の参照は 0、`arm_pose` 0.2） | 53 / 14 |
| `Mjlab-Getup-QddBipedArms` | 方策が動かす | 48 / 14（`policy_io_getup_arms.json`） |

10/18 夜（OTLさん「歩行などでは腕も使うようにして」「干渉とか速度とか変」）に、4 課題とも次のように変えた。

- **腕を方策に入れた**: 歩行・不整地・ジャンプも 14 関節を動かす（それまでは腕を PD で 0 に固定）。
- **モーターの速度とトルクの特性**: 全関節を `actuator/qdd_motor.py` の torque-speed envelope にした（腕は GIM3510-8）。それまでの起き上がりはトルクの上限で切るだけで、実機では出ない速さで振り回せた。電源電圧と強さの乱数化はジャンプと同じ。
- **自分どうしの当たり判定**: モーターの外形と胴の上を、床だけでなくロボット自身とも当てる（`GETUP_SELF_CONTYPE`）。それまでは脚どうし・腕と脚がモーターをすり抜けていた（作り直す前の転倒ライブラリでは、左右の股のモーターが 36% の状態で重なっていた）。
- **歩くときの腕の手本**（10/18 夜、OTLさん「肘を前方に曲げたほうがいい。何かリファレンスを与えたほうがよい」）: 歩行・不整地・ジャンプの腕の既定姿勢を肩 +0.15 rad（少し後ろ。10/18 OTLさん「肩は少しうしろにして」）・肘 1.0 rad（前に 57°）にした。歩行と不整地には `arm_swing`（`mdp.qdd_arm_swing_tracking`、重み 1.0、std 0.25 rad）を足した。手本は歩みの拍子から作り、肩はそれぞれ自分の側の脚と逆向きに振る（左足が着く時に左腕がいちばん後ろ）。振れ幅は前進の指令に比例し、0.5 m/s で ±0.35 rad、後ろ歩きでは逆向き、止まっている時は既定姿勢。腕が前にある時は肘をさらに 0.2 rad 曲げる。実際の脚の動きではなく拍子に合わせるので、脚を動かさずに稼ぐことはできない。姿勢の画像: `/mnt/project-files/robo/arms_walk/arm_swing_reference_v3.png`。
- **肘の向き**（10/18 夜、OTLさん「腕の前後逆にして」）: 肘の軸が +y で、肘 + が前腕を後ろに曲げていた（探索の `gpu_cem.py` も同じ）。軸を −y にして、肘 + で前（+x）に曲がるようにした。肩は − が前のまま。それより前の腕つきの学習（arms2_getup、arms3_*）は後ろ曲がりの肘で学習しているので使わない。転倒姿勢集は作り直した。手本（getup_demo_traj_arms.npz）は肘が 0.2 rad 以下なので作り直していない。
- **転倒ライブラリを作り直した**: 自分どうしの当たり判定ありで作り直し、手本に沿った path の状態は体がめり込むもの（20 cm の手本の前腕が 25 cm の腕では腿に入る）を捨てる。そのため path は寝た姿勢側が減った（tilt の中央値 53° → 42°）。寝た姿勢は bridge_supine と本物の転倒で補う。
- **kd を関節の減衰に移した**（5 ms の物理）: envelope の PD は陽的なので、軽い前腕と宙に浮いた足首が 15 rad/s で小刻みに震えた。kd を MuJoCo の関節減衰（陰的に積分される）にすると止まる。ブレーキ側は envelope でも上限いっぱいまで出せるので、トルクは同じ。ジャンプ（2.5 ms）は電流の計算がアクチュエータのトルクを読むので脚は今のまま。

- 関節の並びは脚 10 個のあと `l_shoulder_pitch, l_elbow, r_shoulder_pitch, r_elbow`。どれもピッチ軸なので、左右反転は入れ替えだけで符号は変えない。
- 起き上がりの転倒姿勢は `scripts/qdd_getup_states.py --arms`（`getup_states_arms.npz`）、手本は探索で見つけた仰向けからの起き上がり（`scripts/qdd_getup_search/export_arms_demo.py` → `getup_demo_traj_arms.npz`）。うつ伏せの手本はない（腕なしの手本は腕があると立てない）ので、うつ伏せの橋渡しの開始姿勢も使わない。
- 腕を下に垂らしたまま既定姿勢で立たせると、胴体が後ろに約 8.7° 傾く（腕なしは 4.5°、20 cm の腕で測った値）。方策が直す。

### 仰向けからの起き上がり（見つけた動き＋修正、10/19）

OTLさんの方針（10/19 朝）: うつ伏せからは寝返りで仰向けになってから起き上がる。仰向けからの学習は続ける。

- arms4/5_getup（手本は報酬だけ）は、本当に倒れた姿勢から一度も立てなかった。
- 新しい課題 `Mjlab-Getup-Supine-QddBipedArms`（`tasks/qdd_biped_arms.py`）は、探索で見つけた関節目標の列（`getup_demo_traj_arms.npz` の `ref_keys` / `ref_durs_s` / `ref_prep_s`、sd010 の結果、探索モデルで 29/32）をそのまま流し、方策はその上に修正（0.5 rad × 行動）を足すだけにした（`mdp.GetupReferenceJointPositionAction`）。
  - 観測は 50 次元で、腕つき起き上がりの 48 次元に、手本の時計 `[sin(pi s), cos(pi s)]` を足したもの。
  - うつ伏せの開始姿勢は使わない（`supine_only`）。
  - policy_io には手本そのもの（`reference`）が入る。実機のコントローラは、同じ時計で手本を流して修正を足す。
- 学習用のシミュレーションで修正なし（`scripts/eval_qdd_getup_supine.py zero`）で流した結果:
  - 寝た姿勢からは、上体が起きてしゃがんだところ（2.4〜3.3 秒、傾き 10° 以内）まで行くが、最後に立ち上がるところで後ろに倒れて、立てたのは 0。
  - 手本の途中からは 50% 立てた。
  - 修正で最後のバランスを直すのが学習の役目。
- 寝返り（うつ伏せ → 仰向け）は `gpu_cem.py --face prone --goal supine` で探す。

### 膝を伸ばし気味に歩く（10/19）

OTLさん「歩行でなるべく膝を曲げない歩き方にしたい」への対応。腕つきの歩行・不整地（`tasks/qdd_biped_arms.py` の `_set_walk_legs`）で、既定の姿勢を変えた。

- 既定の姿勢は膝 0.4 rad（23°。前は 45°）で、股・足首は −0.2 rad にして足裏を平らにした。これが行動の基準、リセットの姿勢、姿勢報酬の中心になる。
- 胴の高さは 0.3688 m（前は 0.3542 m）。
- 歩いているときの膝の姿勢カーネルを 0.4 → 0.25 rad に絞った。
- policy_io の `default_joint_pos` も膝 0.4 になる。起き上がりは従来の 45° で終わるので、歩行に渡すときは方策ごとの既定姿勢に従う。
- 比べた画像: `/mnt/project-files/robo/arms_walk/walk_knee_pose.png`
