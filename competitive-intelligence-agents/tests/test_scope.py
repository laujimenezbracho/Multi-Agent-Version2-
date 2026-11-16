from ci_agents.scope import scope_contract
from ci_agents.agents.planner import _validate_plan,PlannerOutputError
from ci_agents.schemas import ResearchPlan,ResearchTask
from datetime import datetime,timezone
import pytest


def test_list_price_and_total_cost_are_different_requests():
    assert scope_contract('Which competitor has the lowest current Growth plan price?')['kind']=='published_list_price'
    assert scope_contract('Which competitor has the lowest total cost including AI add-ons?')['kind']=='question_defined'
    assert scope_contract('Compare competitors on value and cheapest price')['kind']=='question_defined'


def test_optional_feature_dossier_cannot_be_essential_for_list_price():
    q='Which vendor has the cheapest Growth price?';cutoff=datetime(2026,11,11,tzinfo=timezone.utc)
    p=ResearchPlan(original_question=q,as_of=cutoff,objective='Compare prices',
        tasks=[ResearchTask(task_id='T1',research_question='Compare list prices',companies=['Example'],categories=['Pricing'],time_scope='At cutoff',evidence_requirements=['Price basis']),
            ResearchTask(task_id='T2',research_question='Inspect features',companies=['Example'],categories=['Product launch'],time_scope='At cutoff',evidence_requirements=['Feature matrix'])],
        expected_answer_elements=['Comparable list prices'],comparison_dimensions=['Price'])
    with pytest.raises(PlannerOutputError,match='non-pricing enrichment'):
        _validate_plan(p,q,cutoff,{'Example'},{'Pricing','Product launch'})
    p.tasks[1].priority='optional'
    assert _validate_plan(p,q,cutoff,{'Example'},{'Pricing','Product launch'})==p


def test_price_verification_can_use_promotion_and_product_metadata():
    q='Which vendor has the cheapest Growth price?';cutoff=datetime(2026,11,11,tzinfo=timezone.utc)
    p=ResearchPlan(original_question=q,as_of=cutoff,objective='Verify current prices',
        tasks=[ResearchTask(task_id='T1',research_question='Resolve current price conflicts',
            companies=['Example'],categories=['Pricing','Promotion','Product launch'],
            time_scope='At cutoff',evidence_requirements=['Price basis and effective date'])],
        expected_answer_elements=['Comparable current list prices'],comparison_dimensions=['Price'])
    assert _validate_plan(p,q,cutoff,{'Example'},{'Pricing','Promotion','Product launch'})==p
