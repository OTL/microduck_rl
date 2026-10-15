# QDD 二足の「技」（Mjlab-Skill-QddBiped）

立ち姿勢から始まり立ち姿勢で終わる短い全身の動き（お辞儀など）を、キーフレームの JSON から学習する課題。言葉から技を覚える構想（ロボットのリポジトリ `docs/skill-from-language.md`）の段階 1。あとで LLM が書くのはこの JSON だけで、コードは書かない。

## ファイル

| ファイル | 内容 |
|---|---|
| `src/mjlab_microduck/robot/qdd_biped/skills/skill_spec.schema.json` | 技の仕様の JSON Schema（形だけ。LLM の出力形式に使う） |
| `src/mjlab_microduck/robot/qdd_biped/skills/bow.json` | 手で書いたお辞儀 |
| `src/mjlab_microduck/robot/qdd_biped/skills/warm_start/walk_gait2.onnx` | 初期値に使う歩行の方策（gait2） |
| `src/mjlab_microduck/tasks/qdd_skill.py` | 仕様の読み込み・検証（関門）・参照軌道の生成 |
| `src/mjlab_microduck/tasks/qdd_biped_skill_env_cfg.py` | 課題の設定、PPO の設定、技用の policy_io |
| `src/mjlab_microduck/tasks/mdp.py`（末尾） | 技のクロックと参照（`SkillClockCommand`）、報酬、終了条件 |
| `tests/test_qdd_skill.py` | 検証（良い例・悪い例）、参照、環境、policy_io、初期値の読み込み |

## 技の仕様（schema_version 1）

```json
{
  "schema_version": 1, "name": "bow", "description": "...",
  "duration_s": 3.0, "settle_s": 1.0, "interpolation": "minjerk", "support": "both_feet",
  "keyframes": [
    {"t": 0.0},
    {"t": 1.2, "joints_deg": {"hip_pitch": -55, "knee": 60, "ankle_pitch": -30}, "torso": {"pitch_deg": 25}},
    {"t": 1.9, "joints_deg": {"hip_pitch": -55, "knee": 60, "ankle_pitch": -30}},
    {"t": 3.0}
  ],
  "reward_weights": {"joint_tracking": 3.0}
}
```

- 関節角は絶対値（度）。書かない関節は立ち姿勢の値（hip_pitch -22.5、knee 45、ankle_pitch -22.5、ほか 0）。`hip_pitch` のように左右を付けない名前は両脚に左右対称で入る（yaw・roll は右で符号が逆）。
- 胴体の傾き・高さは関節角から計算する（両足の裏が床に平らに着いているとして順運動学）。`torso` は書き手の意図の確認用で、計算と 5 度・2 cm 以上ずれたら不合格。前傾（お辞儀）が pitch の正。
- 最初（t = 0）と最後（t = duration_s）のキーフレームは立ち姿勢でなければならない。そのあと `settle_s` の間、立ち姿勢に戻って止まることに報酬を出す。
- 報酬の重みは決まった範囲の中だけで変えられる（`qdd_skill.REWARD_WEIGHT_BOUNDS`）。バランス・両足の接地・最後の立ち姿勢は 0 にできない。
- 安全に関わるもの（終了条件、関節と膝の制限、ドメインランダム化、観測と行動の形式）は雛形で固定。仕様に `terminations` などを書くと不合格。

### 検証（LLM の出力が通る関門）

`python -m mjlab_microduck.tasks.qdd_skill <file.json | 名前>`（不合格なら終了コード 1、問題を全部、場所付きで出す）。調べること:

- 形（未知のキー、型、範囲、名前、時刻の順序と間隔 0.2 秒以上、最初と最後が立ち姿勢）
- 関節範囲（3 度の余裕）と姿勢で変わる膝の上限 `knee_max(hip_pitch)`（キーフレームと、その間の補間も）
- 参照の速さ: 関節 2.5 rad/s 以下、胴体の傾きの速さ 1.5 rad/s 以下
- 胴体: 前傾 -15〜30 度（コントローラは 35 度＋角速度、50 度で止める）、横傾き 10 度以内、高さ 0.24 m 以上
- 両足: 足裏が平行（3 度以内）で同じ高さ（1 cm 以内）、左右の間隔 0.08〜0.25 m
- 静的なバランス: 重心が両足の支持多角形の内側に 1.5 cm 以上

## 課題の中身

- 雛形は歩行課題そのもの（ロボット、アクチュエータ、ドメインランダム化、観測のノイズと遅れ、終了条件）。違いは下のとおり。
- **観測は歩行と同じ 41 次元**。指令の欄（36:39）は常に 0、歩容の位相の欄（39:41）に技の位相 `[sin(pi s), cos(pi s)]`（s = 経過時間 / (duration + settle)、0〜1）。
- エピソード = 技 1 回（duration + settle）。立ち姿勢から始める。学習では 30% のリセットを参照軌道の途中の姿勢から始める（reference state initialisation）。
- 報酬: 関節の追従、胴体の傾き・高さの追従、バランス（重心が流れない・回らない）、両足の接地、足の滑り、最後の立ち姿勢、行動の変化・トルク。固定: 関節・膝の制限、自己干渉、足裏を平らに、転倒の罰。
- 終了: 傾き 45 度（歩行は 60 度）、胴体 0.20 m 以下、足以外の接地、膝の制限、NaN。時間切れは技の終わり。
- 押し: ±0.15 m/s（歩行は ±0.3）。左右反転の損失は使わない（技は左右対称とは限らない）。
- 再生（play）: 歩行と同じ名目どおりのロボット、立ち姿勢から技を繰り返す。
- どの技を学習するか: 環境変数 `QDD_SKILL_SPEC`（パスか skills/ の名前）。無ければ `bow`。yatai は環境変数を渡せないので、yatai では既定の技（ブランチの bow.json）を学習する。

## 歩行の方策から始める（warm start）

yatai のハンドラは毎回まっさらに `train` を回し、チェックポイント（.pt）を返さないので、mjlab の `--agent.resume` は使えない。代わりに、技の runner（`QddSkillOnPolicyRunner`）が作られた直後に、**歩行の ONNX から actor の重みと観測の正規化**をそのまま読み込む（ONNX の中の名前が rsl_rl の `mlp.*` / `obs_normalizer` と同じなので、形が合えば写すだけ）。critic は新しく作る。行動のばらつきの初期値は 0.4（ゼロからなら 0.6）。

- 既定の初期値: `skills/warm_start/walk_gait2.onnx`。`QDD_SKILL_INIT_ONNX=<path>` で変更、`none` で無効。
- 歩行の方策は、指令 0 と技の位相を入れても立ったまま転ばない（確認済み）ので、バランスを保てるところから学習が始まる。

## 方策の入出力仕様（技）

技の ONNX には `policy_io`（技用）・`policy_io_sha256`・`skill_spec`（学習に使った JSON）を埋め込む。中身は歩行の policy_io.json と同じ（観測・行動・ゲイン）で、違いは:

- `"kind": "skill"` と `"skill"` 節（名前、duration_s、settle_s、total_s、n_steps、spec_sha256、開始と終了の条件）
- `gait_clock` を「技のクロック」として書いてある: `period_s = 2 * total_s`、`active_threshold = -1`（指令が 0 でも常に進む）、`initial_phase = 1 - dt / (2 * total_s)`（最初の観測で位相 0）。今の Rust コントローラはこの形をそのまま読めるので、`--mode walk --cmd 0,0,0` で技の方策を動かせる。
- `commands` の範囲は全部 [0, 0]。

技は total_s（= n_steps 回）で終わり、その後は学習していない（位相が 1 を越える）。止めるのはコントローラの役目。
