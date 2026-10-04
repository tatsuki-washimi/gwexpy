# リリース成果物を一度だけbuildして昇格する設計

Status: 方針承認済み、spec review approved、ユーザーレビュー待ち、実装未着手。

対象読者は、GWexpyのリリース担当者とrelease workflowの保守担当者である。

この設計は、v0.2.5より後のリリースから適用する。

## 目的

candidateとして検証したsdistとwheelを、そのままGitHub ReleaseとPyPIへ公開する。

現在の`.github/workflows/publish-release.yml`は、手動dispatchとtag pushの両方でpackageをbuildする。

そのため、同じcommitから作ったarchiveでもmetadataが変わるとSHA-256が変わり、candidateで確認したbytesと公開bytesの同一性を示せない。

v0.2.5ではcandidateとtag-runのarchive member名および内容は一致したが、archive metadataの差で配布物のSHA-256が異なった。

v0.2.5の公開物はGitHub Release、PyPI、qualification evidence間のhashが一致しているため、現在のtag、Release、PyPI配布物は変更しない。

## 用語

- **S**：独立レビューとrelease-owner source approvalを受けたsource commit。
- **R**：Sから許可されたrelease evidenceだけを加えたrelease commit。
- **candidate run**：Rからsdistとwheelを一度だけbuildし、その同じbytesでrelease gateを実行するGitHub Actionsの`workflow_dispatch` run。
- **promotion manifest**：candidate run、R、release gate evidence、配布物およびsidecarのSHA-256を結び付けるJSON文書。
- **release GO**：release ownerが特定のRとcandidate artifact一式の公開を明示的に許可する独立した決定。
- **promotion**：candidate runに保存された検証済みbytesをtag-runが取得し、GitHub ReleaseとPyPIへ渡す処理。

## 設計範囲

この変更は`publish-release.yml`、release payload検証script、release workflow contract tests、release手順書、およびrelease contractの設定を対象とする。

package source、配布形式、既存のqualification内容、versioning policy、Trusted Publisher設定は変更しない。

v0.2.5の公開物を再build、置換、yankしない。

## release flow

### candidate run

release担当者は、既存のsource approvalを完了した後、`main`から`release_ref=R`、`expected_tag=vX.Y.Z`を指定してworkflowをdispatchする。

dispatch runは現行のsource検証、protected-branch tip検証、source approval確認を行い、`R`が期待versionとtagに対応することを確認する。

build jobは`workflow_dispatch`でのみ実行し、sdist、wheel、`distribution-sha256.json`、`LICENSE.sha256`を一度だけ生成する。

smoke test、全qualification lane、historical gate、および集約evidence jobは、build jobが作成した同じartifactをdownloadして検査する。

qualification jobはcheckoutしたsourceをtest harnessとして使い、インストール対象のpackageはcandidate artifactから取得する。

すべてのrequired gateがsuccessし、各aggregate evidenceを検証した後にmanifest jobを実行する。

manifest jobはcandidate artifactとsidecarを再hashし、既存の`distribution-sha256.json`の内容とも照合してからpromotion manifestを作る。

manifest jobはpackage、sidecar、promotion manifest、およびmanifestが列挙するrequired gate evidenceをrun IDに結び付けてuploadする。

manifest jobはartifact IDを記録するためにcurrent runのActions artifact一覧をread-onlyで取得する。

payloadとsidecarのretentionは現行の90日を維持する。

candidate artifactが期限切れまたは取得不能になった場合、同じrunを別bytesで補修しない。

release ownerは新しいcandidate runを完了させ、そこから新しいrelease GOを記録する。

### promotion manifest

promotion manifestはUTF-8のJSONとし、sorted keys、duplicate keyなし、末尾LFありの一意なserializationを使う。

manifest自身のSHA-256はファイルのraw bytesから計算し、manifest内には自己digestを含めない。

manifestには`schema=gwexpy-release-promotion-v1`を含め、少なくとも次の情報を記録する。

- schema version、repository、package version、expected tag。
- SとRのfull commit SHA、candidate run IDとattempt番号、event、dispatch ref、workflow ID、canonical workflow path、workflow ref、workflow SHA。
- review evidenceのpathとSHA-256、release noteのpathとSHA-256。
- release contractの識別子とSHA-256、およびversionに適用されるrequired gate一覧。
- 各required jobの結論と、各required aggregate evidence artifactの名前およびSHA-256。
- sdistとwheelの名前、kind、byte size、SHA-256。
- `distribution-sha256.json`と`LICENSE.sha256`の名前およびSHA-256。
- candidate artifactのActions artifact nameとartifact ID。

manifest jobはrequired gateが欠落、skipped、またはsuccess以外の場合にmanifestを作らない。

manifest jobは初回attemptだけを受け付け、`run_attempt=1`を記録する。

失敗したrunをrerunしてattempt番号が増えた場合、そのrun IDはpromotion対象に使わず、新しい`workflow_dispatch`を開始する。

required gate集合はrelease contractから決める。

version別gateの条件分岐を許すが、manifestに記録されたgate集合はtag-runがrelease contractから独立に再計算した集合と完全一致しなければならない。

### release GO

release GOはsource approvalおよびcandidate qualificationとは別のrelease-owner決定として維持する。

release ownerは、release tracking issueのcommentにcanonicalなGO recordを記録する。

GO comment bodyは次の固定形式とし、issue comment APIで得たauthor、issue、作成時刻も検証する。

```text
GWEXPY-RELEASE-GO-v1
version=vX.Y.Z
source_sha=<40 lowercase hexadecimal characters>
candidate_run_id=<decimal run ID>
promotion_manifest_sha256=<64 lowercase hexadecimal characters>
sdist_sha256=<64 lowercase hexadecimal characters>
wheel_sha256=<64 lowercase hexadecimal characters>
decision=GO
```

commentのissueはrelease contractで指定されたrelease tracking issueと一致しなければならない。

commentのauthorはrelease contractで指定されたrelease ownerと一致し、`updated_at`は`created_at`と同じでなければならない。

commentはcandidate runの全required gateがsuccessした時刻より後に作成されていなければならない。

GO recordのissue番号はrelease contractでversionごとに指定する。

tag-runはGO recordをread-onlyで取得し、tag、manifest、R、両distribution digestの全値が一致することを確認する。

source approvalまたはcandidate gate successだけからrelease GOを推定しない。

tag annotationはcandidate runを参照するものであり、それ自体をrelease GOとして扱わない。

### annotated tag binding

release ownerはGO recordの記録後、Rを指すannotated tagを作成する。

tag message bodyは次の固定順序による`GWEXPY-PROMOTION-v1` recordとする。

```text
GWEXPY-PROMOTION-v1
repository=tatsuki-washimi/gwexpy
tag=vX.Y.Z
source_sha=<40 lowercase hexadecimal characters>
candidate_run_id=<decimal run ID>
promotion_manifest_sha256=<64 lowercase hexadecimal characters>
release_go_comment_id=<decimal comment ID>
```

validatorはUTF-8、LF改行、行順、field set、値の形式を検証し、重複field、未知field、追加行を拒否する。

workflowは`release_go_comment_id`のcommentをrelease contractで指定したissueから取得し、GO recordの全値がtag bindingとmanifestに一致することを検証する。

validatorはannotated tag objectを要求し、tagが指すcommit、tag名、record内のrepository、tag名、source SHAの一致を検証する。

### tag-triggered promotion

tag pushはpublication workflowを起動し、build jobおよびcandidate qualification jobを実行しない。

candidate manifest jobは`contents: read`と`actions: read`だけを持つ。

promotion verification jobは`contents: read`、`actions: read`、およびGO record取得に必要な`issues: read`だけを持つ。

promotion verification jobはtag recordのrun IDを使ってcandidate run metadataとそのrunに属するartifactを取得する。

candidate runは同じrepositoryの成功済み`workflow_dispatch`であり、dispatch refが`refs/heads/main`、source SHAがtag peel SHA、versionとtagがmanifestに記録された値と一致しなければならない。

candidateのworkflow SHAはtagから起動したpromotion workflowのworkflow SHAおよびRと一致しなければならない。

GitHub Actions run APIの`workflow_id`と`path`は、release contractで固定した`.github/workflows/publish-release.yml`のIDとpathに一致しなければならない。

runの`run_attempt`とmanifestのattempt番号は1で一致しなければならない。

promotion verification jobはcandidate runの全required job status、manifest hash、aggregate evidence hash、package hash、sidecar hash、release note hashを照合する。

candidate runのartifactを別のrun、branch、tag、workflowから補完しない。

GitHub Release jobとPyPI jobはそれぞれ`actions: read`を持ち、candidate run IDからartifactをdownloadしてmanifestに対してpackageとsidecarを再検証する。

GitHub Actionsのjob間でrunner filesystemを共有せず、各publisher jobが同じcandidate run IDを指定してartifactを取得する。

すべてのidentity checkとhash checkが通った後に、GitHub Release jobが検証済みcandidate bytesを使ってReleaseを作成する。

GitHub Releaseにはsdist、wheel、`distribution-sha256.json`、`LICENSE.sha256`、promotion manifestを添付する。

Release notesはR内の検証済みrelease note bytesを使用し、そのSHA-256もmanifestとの一致を確認する。

GitHub Release readbackはtag target、notes、添付asset名、各assetのbytesおよびSHA-256を検証する。

PyPI jobはGitHub Release readback成功後にのみ実行し、同じcandidate run IDから取得したsdistとwheelを再検証して公開する。

PyPI readback jobはpublish job成功後に実行し、PyPI JSON APIからrelease versionのfilenameとSHA-256を取得する。

readback jobはAPIに記録されたURLからsdistとwheelのbytesをdownloadしてSHA-256を再計算し、promotion manifestおよびGitHub Release assetの値と照合する。

readback jobは一時的な404、5xx、network errorを最大15分間retryする。

version、filename、またはSHA-256の不一致はretryせずfailする。

PyPI metadata、filename、またはdownloaded bytesのいずれかが不一致か取得不能ならrelease closureをfailとし、release完了を宣言しない。

GitHub Release作成jobとPyPI jobは別jobのまま維持し、それぞれ必要な最小権限を持つ。

workflow内の外部actionはfull commit SHAでpinする。

## 失敗時の動作

tag format、GO record、run metadata、required job status、manifest、evidence、package、sidecarのいずれかが不一致なら、GitHub Release作成前にfailする。

tag-runでbuildやrebuildへfallbackしない。

candidate artifactが期限切れ、削除済み、またはdownload不可の場合、そのtagからpublicationを続けない。

tag作成前にcandidate artifactが失われた場合は、新しいcandidate runと新しいGOを作る。

GitHub Releaseが既に存在する再実行では、既存Releaseのtarget、notes、asset bytesがcandidateと完全一致する場合に限りreadback済みとして後続jobへ進む。

既存Releaseの内容が異なる場合は停止し、assetの置換や同versionの再buildを行わない。

PyPI publishまたはreadbackが失敗した場合はrelease acceptanceをHOLDにする。

部分upload時はPyPI version JSONのfile一覧とSHA-256を読み、candidate manifestおよび保存された同一run payloadと照合する。

filenameの追加、hashの不一致、またはartifact identityの不確実さがあれば復旧を止め、調査する。

片方のdistributionだけが期待hashで存在する場合、release ownerが明示的にreviewして許可した後に限り、同一candidate runの検証済みpayloadから欠けたdistributionだけをuploadする。

strict publish action全体を盲目的に再実行せず、欠落distributionをrebuildせず、`skip-existing`を使わない。

復旧後はPyPIにあるwheelとsdistの両方をreadbackし、期待hashとの一致を記録する。

tag作成後にcandidate artifactが取得できなくなった場合や、tagが不正なcandidateを指している場合は、tagや公開物を自動で移動、置換、削除しない。

## 検証要件

`tests/test_publish_release_workflow.py`と`tests/test_release_gate_workflow_contract.py`は、candidate buildがdispatch経路だけで一度実行され、tag経路にbuild jobがないことを検証する。

contract testsは、全qualification jobがcandidate artifactの同じdistribution SHA-256を検証すること、manifestがrequired gatesとevidence digestを拘束すること、tag recordがcandidate runとmanifest digestを指定することを検証する。

validatorのテストは、誤ったrepository、event、ref、workflow SHA、source SHA、version、run ID、manifest SHA、distribution SHA、missing artifact、expired artifact、欠落gate、失敗gate、skipped gate、GO不一致を拒否することを確認する。

workflow contract testsは、拒否ケースでGitHub Release作成とPyPI publishが実行されないことを確認する。

workflow contract testsは、別workflow IDまたはpath、run attemptが1以外のcandidate runを拒否することも確認する。

readback testsは、GitHub Release assetとPyPI API metadataおよびdownloaded package bytesのSHA-256がpromotion manifest内のcandidate package SHA-256と一致することを確認する。

権限contract testsは、promotion verificationとpublisher jobsがcross-run artifact取得に必要な`actions: read`だけを持ち、GitHub Releaseの作成とPyPI OIDC publishが別jobに分離されていることを確認する。

権限contract testsはPyPI readback jobがpackageをread-onlyで取得し、PyPI credentialsを持たないことを確認する。

## 完了条件

promotion manifest、annotated tag、GitHub Release、PyPI metadata、PyPI downloaded bytesの各readbackが同じR、candidate run ID、sdist SHA-256、wheel SHA-256に結び付く。

tag-runはcandidate artifactを取得して公開し、package buildを実行しない。

失敗時は検証境界より先へ進まず、別runや別bytesから公開を継続しない。

release ownerのGOはcandidate manifestの具体値に結び付き、source approvalや技術qualificationと混同されない。

## 実装時に参照する既存ファイル

- `.github/workflows/publish-release.yml`：candidate、qualification、GitHub Release、PyPIの現行job graph。
- `scripts/ci/validate_release_payload.py`：distribution manifestとpackage hashの検証。
- `scripts/ci/assemble_release_evidence.py`：現行の同一run integration evidence作成。
- `scripts/ci/verify_release_human_approval.py`：source approval確認。release GO確認とは別の責任を保つ。
- `scripts/ci/release_contracts.json`：version別release contract。
- `tests/test_publish_release_workflow.py`：workflow behavior contract。
- `tests/test_release_gate_workflow_contract.py`：release gate contract。
- `docs/developers/plans/20260927_v0.2.5_release_plan.md`：S、R、approval、exact-candidate qualificationの既存用語と境界。
