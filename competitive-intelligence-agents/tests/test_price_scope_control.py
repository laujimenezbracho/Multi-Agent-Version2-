"""Simple list-price questions must not become checkout or TCO investigations."""
from datetime import datetime,timezone
import pytest
from ci_agents.schemas import ResearchPlan,ResearchTask
from ci_agents.scope import normalize_price_plan

COMPANIES=['ClearPilot','NexaFlow','OrbitDesk','PulseGrid','VertexOps']


def plan(question='Which competitor has the lowest current Growth plan price?'):
    return ResearchPlan(original_question=question,as_of=datetime(2026,11,11,tzinfo=timezone.utc),
        objective='Capture every checkout and page state before comparing prices',
        tasks=[ResearchTask(task_id='T1',research_question='Gather prices and checkout screenshots',
            companies=COMPANIES,categories=['Pricing','Promotion'],time_scope='At cutoff',
            evidence_requirements=['Published Growth price','Exact page UI currency state','Mandatory checkout fees and seat minima'])],
        expected_answer_elements=['All page UI states and verified checkout fees'],comparison_dimensions=['All-in cost'])


def test_simple_price_plan_is_bounded_without_changing_identity():
    original=plan();snapshot=original.model_dump_json()
    result=normalize_price_plan(original,COMPANIES)
    assert result.original_question==original.original_question and result.as_of==original.as_of
    assert len(result.tasks)==1 and result.tasks[0].companies==sorted(COMPANIES)
    assert result.tasks[0].categories==['Pricing']
    assert len(result.tasks[0].evidence_requirements)==2
    assert result.optional_context and 'checkout' in ' '.join(result.optional_context).lower()
    assert original.model_dump_json()==snapshot


@pytest.mark.parametrize('question',[
    'Which competitor has the lowest total cost including fees?',
    'Which competitor has the lowest Growth price including taxes?',
    'Which competitor has the lowest Growth price in the UK?',
    'What are the top competitive threats for PulseGrid?',
])
def test_explicit_extended_scope_is_preserved(question):
    original=plan(question)
    assert normalize_price_plan(original,COMPANIES)==original


def test_explicit_competitor_subset_is_preserved():
    original=plan('Which competitor has the lowest Growth price: NexaFlow or OrbitDesk?')
    result=normalize_price_plan(original,COMPANIES)
    assert result.tasks[0].companies==['NexaFlow','OrbitDesk']
