"""Offline V2 tests. All stage functions are mocked; no live baseline is executed."""
import asyncio
from dataclasses import replace
from datetime import datetime, timezone, timedelta
import json
from pathlib import Path
from types import SimpleNamespace
from unittest.mock import AsyncMock

from agents import Usage
import httpx
try:
    import httpx2
except ImportError:
    httpx2 = httpx
import pytest

from ci_agents import pipeline, pipeline_v2 as v2, cli
from ci_agents.agents.reviewer import report_fingerprint
from ci_agents.config import load_config
from ci_agents.dataset import load_dataset
from ci_agents.persistence import verify_manifest
from ci_agents.recovery import select_action, merge_evidence
from ci_agents.schemas import (ResearchPlan, ResearchTask, EvidenceItem, EvidencePack, ReportClaim,
    CompetitiveReport, ReviewIssue, ReviewResult, TaskCoverage, RunResult)
from ci_agents.telemetry import observe_sdk_result, observe_retrieval_event
from ci_agents.v2_baseline import load_baseline

PUBLIC = Path(__file__).resolve().parents[1] / "data/public"
CUTOFF = datetime(2026, 11, 11, 23, 59, 59, tzinfo=timezone.utc)
QUESTION = "Compare supplied product observations"


def measured(incoming=100, outgoing=20):
    observe_sdk_result(SimpleNamespace(context_wrapper=SimpleNamespace(usage=Usage(
        requests=1, input_tokens=incoming, output_tokens=outgoing, total_tokens=incoming+outgoing))))


def evidence_item(row):
    fields = {name: row.get(name) for name in ("record_id", "claim_id", "company_name", "source_type", "source_url",
        "published_at", "available_at", "effective_date", "end_date", "source_independence_group", "supersedes_record_id",
        "verification_status", "feature_status")}
    return EvidenceItem(**fields, source_title=row["document_title"], finding=row["summary"],
                        quoted_or_paraphrased_evidence=row["details"], confidence=0.8)


def issue(kind="missing_required_information", id="I1", severity="major", missing=None):
    return ReviewIssue(issue_id=id, severity=severity, issue_type=kind, affected_claim_ids=["C1"],
        related_record_ids=["WEB-00001"], missing_record_ids=missing or [],
        explanation="A required evidence component is missing", suggested_correction="Address this specific evidence gap")


def make_review(plan, report, decision, issues):
    return ReviewResult(decision=decision, completeness_score=1 if decision=="PASS" else 0.5,
        evidence_support_score=1, temporal_consistency_score=1, flagged_issues=issues,
        reviewer_notes=["Synthetic test review"], reviewed_question=plan.original_question,
        reviewed_report_hash=report_fingerprint(report))


@pytest.fixture(autouse=True)
def offline(monkeypatch):
    mocks=[]
    for library in {httpx,httpx2}:
        blocked=AsyncMock(side_effect=AssertionError("HTTP prohibited"))
        monkeypatch.setattr(library.AsyncClient,"send",blocked)
        monkeypatch.setattr(library.Client,"send",lambda *a,**k:pytest.fail("HTTP prohibited"))
        mocks.append(blocked)
    yield
    for mock in mocks: mock.assert_not_called()


@pytest.fixture
def settings(tmp_path):
    return load_config(environ={"OPENAI_API_KEY":"test-credential", "OPENAI_MODEL":"gpt-5-mini",
        "DATA_DIR":str(PUBLIC), "RUNS_DIR":str(tmp_path/'runs')})


@pytest.fixture
def baseline_factory(settings,monkeypatch):
    records=load_dataset(PUBLIC).records
    state=SimpleNamespace(order=[],decision="FAIL",issues=[issue()],fail=None)
    plan=ResearchPlan(original_question=QUESTION,as_of=CUTOFF,objective="Collect comparable observations",
        tasks=[ResearchTask(task_id="T1",research_question=QUESTION,companies=[records[0]["company_name"]],
                           categories=["Pricing"],time_scope="At cutoff",evidence_requirements=["Public source evidence"])],
        expected_answer_elements=["Answer and limitations"],comparison_dimensions=["Pricing"])
    event={"tool":"get_record","status":"ok","record_ids":[records[0]["record_id"]]}
    pack=EvidencePack(original_question=QUESTION,as_of=CUTOFF,evidence_items=[evidence_item(records[0])],
        retrieval_log=[json.dumps(event)],missing_information=["Initial evidence gap"],
        coverage_by_task=[TaskCoverage(task_id="T1",coverage_score=None,supporting_record_ids=[records[0]["record_id"]])])
    report=CompetitiveReport(question=QUESTION,as_of=CUTOFF,executive_summary="One supplied observation",
        key_findings=["Limited evidence"],competitor_comparison=["Comparison incomplete"],
        strategic_implications=["No supported ranking"],limitations=["Synthetic fixture"],
        claims=[ReportClaim(claim_id="C1",text=records[0]["summary"],claim_type="factual",
                            supporting_record_ids=[records[0]["record_id"]],confidence=0.8)],sources=[records[0]["record_id"]])
    async def planner(*args):
        state.order.append("planner")
        if state.fail=="planner":raise RuntimeError("test-credential")
        measured();return plan.model_copy(deep=True)
    async def researcher(*args):
        state.order.append("researcher");measured();observe_retrieval_event(event);return pack.model_copy(deep=True)
    async def analyst(*args):
        state.order.append("analyst");measured();return report.model_copy(deep=True)
    async def reviewer(p,e,r):
        state.order.append("reviewer");measured();return make_review(p,r,state.decision,state.issues)
    for name,func in [("planner",planner),("researcher",researcher),("analyst",analyst),("reviewer",reviewer)]:
        monkeypatch.setattr(pipeline,name,AsyncMock(side_effect=func))
    def build(run_id="baseline",decision="FAIL",issues=None):
        state.decision=decision;state.issues=issues if issues is not None else ([issue()] if decision=="FAIL" else [])
        result=asyncio.run(pipeline.run_pipeline(QUESTION,CUTOFF,run_id,settings=settings))
        state.order.clear()
        for name in ["planner","researcher","analyst","reviewer"]:getattr(pipeline,name).reset_mock()
        return Path(result.artifact_directory)
    return build,state,records


@pytest.fixture
def recovery_stages(monkeypatch,baseline_factory):
    _,_,records=baseline_factory
    state=SimpleNamespace(order=[],decision="PASS",fail=None,contexts={})
    async def researcher(plan,service,*,limits):
        state.order.append("researcher");state.contexts["researcher"]=plan.model_copy(deep=True)
        measured(10,5)
        if state.fail=="researcher":raise RuntimeError("test-credential")
        row=service.get_record(records[1]["record_id"],as_of=plan.as_of)
        event={"tool":"get_record","status":"ok","record_ids":[row["record_id"]]}
        observe_retrieval_event(event)
        return EvidencePack(original_question=plan.original_question,as_of=plan.as_of,evidence_items=[evidence_item(row)],
            retrieval_log=[json.dumps(event)],coverage_by_task=[TaskCoverage(task_id=task.task_id,coverage_score=None,
                supporting_record_ids=[row["record_id"]]) for task in plan.tasks])
    async def analyst(plan,evidence):
        state.order.append("analyst");state.contexts["analyst"]=plan.model_copy(deep=True)
        measured(10,5)
        if state.fail=="analyst":raise RuntimeError("test-credential")
        ids=[item.record_id for item in evidence.evidence_items]
        return CompetitiveReport(question=plan.original_question,as_of=plan.as_of,executive_summary="Revised supplied observation",
            key_findings=["Evidence-qualified revision"],competitor_comparison=["Limited comparison"],
            strategic_implications=["Uncertainty retained"],limitations=["Mocked revision"],
            claims=[ReportClaim(claim_id="C1",text="Revised statement based on supplied evidence",claim_type="inference",
                                supporting_record_ids=ids,confidence=0.6)],sources=ids)
    async def reviewer(plan,evidence,report):
        state.order.append("reviewer");state.contexts["reviewer"]=plan.model_copy(deep=True)
        measured(10,5)
        if state.fail=="reviewer":raise RuntimeError("test-credential")
        return make_review(plan,report,state.decision,[] if state.decision=="PASS" else [issue("unsupported_factual_claim")])
    for name,func in [("researcher",researcher),("analyst",analyst),("reviewer",reviewer)]:
        monkeypatch.setattr(v2,name,AsyncMock(side_effect=func))
    return state


def paired(settings,directory,run_id="v2-test"):
    return asyncio.run(v2.run_pipeline_v2(baseline_dir=directory,run_id=run_id,settings=settings))


def test_initial_pass_skips_every_recovery_stage(settings,baseline_factory,recovery_stages):
    build,state,_=baseline_factory;directory=build(decision="PASS")
    result=paired(settings,directory)
    assert result.configuration=="v2" and result.execution_status=="completed"
    assert not state.order and not recovery_stages.order
    metrics=json.loads((Path(result.artifact_directory)/'metrics.json').read_text())
    assert metrics['incremental_total']['total_tokens']==0 and metrics['incremental_total']['estimated_cost_usd']==0
    assert metrics['initial_v1']['total_tokens']==480
    assert result.total_metrics.input_tokens==400


@pytest.mark.parametrize("kind,expected",[("missing_required_information",["researcher","analyst","reviewer"]),
    ("unsupported_factual_claim",["analyst","reviewer"]),("weak_strategic_inference",["analyst","reviewer"])])
def test_one_recovery_route_and_telemetry(settings,baseline_factory,recovery_stages,kind,expected):
    build,initial,_=baseline_factory;directory=build(issues=[issue(kind)])
    original={p.relative_to(directory).as_posix():p.read_bytes() for p in directory.rglob("*") if p.is_file()}
    result=paired(settings,directory)
    assert recovery_stages.order==expected and not initial.order
    for name in ['planner','researcher','analyst','reviewer']:getattr(pipeline,name).assert_not_called()
    assert result.review_result.decision=="PASS"
    output=Path(result.artifact_directory)
    meta=json.loads((output/'metadata.json').read_text())
    assert meta['recovery_attempted'] is True and meta['recovery_state']=='completed'
    assert meta['baseline_run_id']=='baseline' and meta['trigger_issue_ids']==['I1']
    assert meta['initial_reviewer_decision']=='FAIL' and meta['final_reviewer_decision']=='PASS'
    totals=json.loads((output/'metrics.json').read_text())
    assert totals['incremental_total']['total_tokens']==15*len(expected)
    assert totals['final_totals']['total_tokens']==480+15*len(expected)
    assert totals['initial_v1_source']=='inherited_archive_not_new_calls'
    assert totals['executed_this_invocation']==totals['incremental_total']
    assert totals['re_review']['llm_requests']==1
    assert result.total_metrics.estimated_cost_usd is None
    assert {p.relative_to(directory).as_posix():p.read_bytes() for p in directory.rglob("*") if p.is_file()}==original
    assert (output/'initial_report.json').read_bytes()==original['report.json']
    assert (output/'initial_review.json').read_bytes()==original['review.json']
    assert json.loads((output/'claim_changes.json').read_text())['modified'][0]['claim_id']=='C1'
    assert 'V2 TARGETED REVISION CONTEXT' in recovery_stages.contexts['analyst'].objective
    assert 'V2 TARGETED' not in recovery_stages.contexts['reviewer'].objective
    assert RunResult.model_validate_json((output/'run_result.json').read_text())==result
    verify_manifest(output)


def test_mixed_issues_use_single_bounded_cycle(settings,baseline_factory,recovery_stages):
    directory=baseline_factory[0](issues=[issue(),issue('unsupported_factual_claim',id='I2')])
    result=paired(settings,directory)
    assert recovery_stages.order==['researcher','analyst','reviewer']
    target=recovery_stages.contexts['researcher']
    assert len(target.tasks)==1 and target.tasks[0].task_id=='REC-I1'
    assert target.as_of==CUTOFF
    assert 'I2' in recovery_stages.contexts['analyst'].objective
    changes=json.loads((Path(result.artifact_directory)/'evidence_changes.json').read_text())
    assert changes['new_evidence_ids']==['WEB-00002']
    v2.researcher.assert_awaited_once();v2.analyst.assert_awaited_once();v2.reviewer.assert_awaited_once()


def test_second_fail_stops_without_another_cycle(settings,baseline_factory,recovery_stages):
    directory=baseline_factory[0]();recovery_stages.decision='FAIL'
    result=paired(settings,directory)
    assert result.execution_status=='completed_review_failed'
    assert recovery_stages.order==['researcher','analyst','reviewer']
    assert json.loads((Path(result.artifact_directory)/'metadata.json').read_text())['recovery_state']=='completed'


def test_unactionable_fail_is_skipped(settings,baseline_factory,recovery_stages):
    directory=baseline_factory[0](issues=[])
    result=paired(settings,directory)
    assert result.execution_status=='completed_review_failed' and not recovery_stages.order
    meta=json.loads((Path(result.artifact_directory)/'metadata.json').read_text())
    assert meta['recovery_reason']=='no_actionable_material_issues' and meta['recovery_attempted'] is False


def test_known_missing_records_need_revision_only(settings,baseline_factory,recovery_stages):
    directory=baseline_factory[0](issues=[issue(missing=['WEB-00001'])])
    paired(settings,directory)
    assert recovery_stages.order==['analyst','reviewer']


@pytest.mark.parametrize('failure',['researcher','analyst','reviewer'])
def test_recovery_exception_persists_partial_outputs_no_retry(settings,baseline_factory,recovery_stages,failure):
    directory=baseline_factory[0]();recovery_stages.fail=failure
    result=paired(settings,directory)
    assert result.execution_status=='execution_failed'
    expected=['researcher','analyst','reviewer'][:['researcher','analyst','reviewer'].index(failure)+1]
    assert recovery_stages.order==expected
    out=Path(result.artifact_directory)
    assert (out/'initial_report.json').exists() and (out/'error.json').exists()
    assert (out/'revised_report.json').exists()==(failure=='reviewer')
    if failure=='reviewer':assert result.review_result is None
    for file in out.iterdir():assert 'test-credential' not in file.read_text()
    verify_manifest(out)


def test_from_scratch_delegates_initial_execution_to_exact_v1(settings,baseline_factory,recovery_stages):
    _,state,_=baseline_factory
    result=asyncio.run(v2.run_pipeline_v2(QUESTION,CUTOFF,'fresh-v2',settings=settings))
    assert state.order==['planner','researcher','analyst','reviewer']
    assert recovery_stages.order==['researcher','analyst','reviewer']
    assert result.execution_status=='completed'
    meta=json.loads((Path(result.artifact_directory)/'metadata.json').read_text())
    assert meta['mode']=='from_scratch' and meta['baseline_run_id'].startswith('v1-initial-')
    assert (settings.runs_dir/meta['baseline_run_id']/'manifest.json').exists()


def test_initial_execution_failure_never_recovers(settings,baseline_factory,recovery_stages):
    _,state,_=baseline_factory;state.fail='planner'
    result=asyncio.run(v2.run_pipeline_v2(QUESTION,CUTOFF,'initial-failed',settings=settings))
    assert result.execution_status=='execution_failed' and not recovery_stages.order
    assert state.order==['planner']
    assert (Path(result.artifact_directory)/'initial_error.json').exists()


@pytest.mark.parametrize('mismatch',['model','cutoff','dataset','prompts'])
def test_baseline_compatibility_rejects_changes(settings,baseline_factory,recovery_stages,monkeypatch,mismatch):
    directory=baseline_factory[0]()
    if mismatch=='model':settings=replace(settings,openai_model='different-model')
    elif mismatch=='cutoff':settings=replace(settings,evaluation_cutoff=CUTOFF-timedelta(days=1))
    else:
        import ci_agents.v2_baseline as loader
        monkeypatch.setattr(loader,'dataset_fingerprints' if mismatch=='dataset' else 'prompt_fingerprints',lambda *args:{'changed':True})
    with pytest.raises(ValueError):paired(settings,directory)
    assert not recovery_stages.order and not (settings.runs_dir/'v2-test').exists()


def test_baseline_tampering_rejected(settings,baseline_factory,recovery_stages):
    directory=baseline_factory[0]();(directory/'plan.json').write_text('{}')
    with pytest.raises(ValueError,match='fingerprint'):paired(settings,directory)
    assert not recovery_stages.order


def test_cannot_chain_v2_or_overwrite_a_v2_run(settings,baseline_factory,recovery_stages):
    directory=baseline_factory[0]();first=paired(settings,directory)
    with pytest.raises(ValueError):paired(settings,Path(first.artifact_directory),'chain')
    count=len(recovery_stages.order)
    with pytest.raises(FileExistsError):paired(settings,directory)
    assert len(recovery_stages.order)==count


def test_private_and_independent_evaluation_files_never_enter_context(settings,baseline_factory,recovery_stages,monkeypatch):
    directory=baseline_factory[0]()
    # An unrelated audit file must never be read, copied, or passed to agents.
    (directory/'evaluation.json').write_text('PRIVATE_EVALUATOR_SENTINEL')
    original=Path.open
    def guard(path,*args,**kwargs):
        assert 'evaluation_private' not in path.parts and not path.name.startswith('private_')
        assert path.name!='evaluation.json'
        return original(path,*args,**kwargs)
    monkeypatch.setattr(Path,'open',guard)
    result=paired(settings,directory)
    for context in recovery_stages.contexts.values():
        assert 'PRIVATE_EVALUATOR_SENTINEL' not in context.model_dump_json()
    assert not (Path(result.artifact_directory)/'initial_evaluation.json').exists()


def test_cli_v2_gates_and_dry_run_without_agents(settings,baseline_factory,recovery_stages,monkeypatch,capsys):
    with pytest.raises(SystemExit):cli.main(['v2','--question','test'])
    directory=baseline_factory[0]()
    monkeypatch.setattr(cli,'load_config',lambda **kwargs:settings)
    assert cli.main(['v2','--baseline-dir',str(directory),'--dry-run'])==0
    assert json.loads(capsys.readouterr().out)['baseline']['baseline_run_id']=='baseline'
    assert not recovery_stages.order
    assert cli.main(['v2','--question-id','Q15','--dry-run'])==0
    assert not recovery_stages.order


def test_fresh_initial_pass_uses_v1_once_and_no_recovery(settings,baseline_factory,recovery_stages):
    _,state,_=baseline_factory;state.decision='PASS';state.issues=[]
    result=asyncio.run(v2.run_pipeline_v2(QUESTION,CUTOFF,'fresh-pass',settings=settings))
    assert state.order==['planner','researcher','analyst','reviewer']
    assert not recovery_stages.order and result.execution_status=='completed'


def test_duplicate_source_merge_is_lossless_and_does_not_mutate_initial(settings,baseline_factory):
    directory=baseline_factory[0]()
    original=load_baseline(directory,settings).evidence
    before=original.model_dump_json()
    added=original.model_copy(deep=True)
    added.evidence_items[0].finding='Additional observation from the same source'
    merged=merge_evidence(original,added)
    assert len(merged.evidence_items)==1
    assert 'Additional observation' in merged.evidence_items[0].finding
    assert original.model_dump_json()==before
    added.evidence_items[0].source_title='Wrong source'
    with pytest.raises(ValueError,match='metadata'):merge_evidence(original,added)


def test_minor_only_fail_is_unactionable(settings,baseline_factory,recovery_stages):
    directory=baseline_factory[0](issues=[issue('weak_strategic_inference',severity='minor')])
    result=paired(settings,directory)
    assert not recovery_stages.order and result.execution_status=='completed_review_failed'


def test_baseline_sdk_version_mismatch(settings,baseline_factory,recovery_stages,monkeypatch):
    directory=baseline_factory[0]()
    import ci_agents.v2_baseline as loader
    monkeypatch.setattr(loader,'version',lambda *args:'different-sdk')
    with pytest.raises(ValueError,match='SDK'):paired(settings,directory)
    assert not recovery_stages.order


def test_never_write_inside_original_baseline(settings,baseline_factory,recovery_stages):
    directory=baseline_factory[0]()
    before={p.relative_to(directory).as_posix():p.read_bytes() for p in directory.rglob("*") if p.is_file()}
    with pytest.raises(ValueError,match='inside'):
        paired(replace(settings,runs_dir=directory),directory)
    assert before=={p.relative_to(directory).as_posix():p.read_bytes() for p in directory.rglob("*") if p.is_file()}
    assert not recovery_stages.order


def test_recovery_researcher_wrong_question_rejected(settings,baseline_factory,recovery_stages,monkeypatch):
    directory=baseline_factory[0]()
    original=v2.researcher
    async def bad(*args,**kwargs):
        output=await original(*args,**kwargs)
        output.original_question='Changed question'
        return output
    monkeypatch.setattr(v2,'researcher',bad)
    result=paired(settings,directory)
    assert result.execution_status=='execution_failed'
    assert recovery_stages.order==['researcher']


def test_cli_live_v2_dispatch_uses_only_mocked_controller(settings,baseline_factory,monkeypatch,capsys):
    directory=baseline_factory[0]()
    monkeypatch.setattr(cli,'load_config',lambda **kwargs:settings)
    mock=AsyncMock(return_value=SimpleNamespace(run_id='mock-v2',execution_status='completed_review_failed',
                                                review_result=SimpleNamespace(decision='FAIL')))
    monkeypatch.setattr(v2,'run_pipeline_v2',mock)
    assert cli.main(['v2','--baseline-dir',str(directory),'--run-id','mock-v2','--allow-api'])==0
    assert mock.call_args.kwargs['baseline_dir']==str(directory)
    assert mock.await_count==1


def test_explicit_manual_only_issue_skips_automated_recovery(settings,baseline_factory,recovery_stages):
    manual=issue('unsupported_factual_claim')
    manual.suggested_correction='Manual review required'
    directory=baseline_factory[0](issues=[manual])
    result=paired(settings,directory)
    assert not recovery_stages.order
    routing=json.loads((Path(result.artifact_directory)/'recovery_routing.json').read_text())
    assert routing['route']=='skip' and routing['ignored_issue_ids']==['I1']
    assert result.execution_status=='completed_review_failed'
