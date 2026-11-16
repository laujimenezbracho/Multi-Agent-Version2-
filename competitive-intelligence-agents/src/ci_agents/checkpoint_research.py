"""Public-only experimental Researcher with resumable, nested evidence checkpoints.

The production Researcher is unchanged. The selector proposes up to four independent
operations; the controller executes only the next checkpoint's prefix and preserves
pending actions. Analyst/Reviewer outputs never enter later research decisions.
"""
from copy import deepcopy
from dataclasses import replace
from importlib.resources import files
import json
from typing import Literal

from agents import Agent, OpenAIResponsesModel, RunConfig, Runner
from openai import AsyncOpenAI
from pydantic import Field

from .agents.researcher import ResearchOutcome, ResearchSession, ResearcherLimits
from .config import OPENAI_ACCEPT_ENCODING, agent_model_settings
from .schemas import Contract, ResearchPlan
from .telemetry import observe_sdk_result

CHECKPOINTS=(8,16,32)
QUESTIONS=('Q01','Q03','Q05','Q07','Q12','Q15')
SELECTOR_INSTRUCTIONS='''You are the Researcher in a controlled research-budget experiment.
Use ONLY the original question, research plan, supplied public search/history responses
and full records. Never access private rubrics or invent facts. Your job is to propose
the NEXT useful public retrieval operations, not answer the question or write evidence.

Choose 1–4 independent operations per decision. A search is discovery, not full evidence.
For search_records set query to a short lexical query (usually 1–3 words); the search
uses AND semantics, so do not paste a long question. Use company/category filters when
helpful. get_pricing_history reveals time/billing comparisons, but every cited source
still requires get_record. For get_record select ONLY record IDs already discovered
in supplied responses or explicit related/superseded IDs. Do not guess record IDs.
Batch independent searches/known-record reads, not actions needing unseen results.
Use null for irrelevant arguments. Every operation costs one retrieval call.

Keep pursuing ESSENTIAL unanswered evidence requirements until they are adequately
covered or no purposeful new operation exists. Resolve current-vs-historical conflicts,
verify alleged events against independent/official sources, and compare relevant
competitors. Read full sources instead of relying on snippets. Price/value statements
need the subject company's baseline too. A lone fresh record per company is rarely
enough for a broad strategic comparison. Avoid repeated identical lookups and irrelevant
enrichment. For pricing history specify the requested tier when applicable.

Set status=no_useful_actions with actions=[] only when you can explain why further
purposeful research is exhausted. Otherwise status=continue with nonempty actions.
The overall cap is 32 operations. Intermediate observation points are hidden from you.
No report, Reviewer feedback, evaluator answer key or score is available to you.
Return only the NextResearchActions structured object. Reason and remaining_gaps are
brief externally reviewable research justifications, not private chain-of-thought.
'''


class ResearchAction(Contract):
    tool:Literal['search_records','get_record','get_pricing_history']
    query:str|None
    company:str|None
    category:str|None
    record_id:str|None
    plan:str|None
    reason:str

    def arguments(self):
        fields={'search_records':['query','company','category'],
                'get_record':['record_id'],'get_pricing_history':['company','plan']}[self.tool]
        return {name:getattr(self,name) for name in fields if getattr(self,name) is not None}


class NextResearchActions(Contract):
    status:Literal['continue','no_useful_actions']
    actions:list[ResearchAction]=Field(max_length=4)
    remaining_gaps:list[str]
    stop_explanation:str


class CheckpointResearcher:
    def __init__(self,plan:ResearchPlan,service,settings,state=None):
        self.plan,self.settings=plan,settings
        self.session=ResearchSession(plan,service,ResearcherLimits(max_tool_calls=32,
            max_search_results=10,max_evidence_items=40,max_full_records=40,max_model_turns=62))
        self.outputs=[];self.pending=[];self.decisions=[];self.stopped=False
        self.stop_explanation='';self.known_ids=set();self.error=None
        if state:
            self.session.calls=state['calls'];self.session.full_calls=state['full_calls']
            self.session.records=state['records'];self.session.log=state['log']
            self.outputs=state['outputs'];self.pending=[ResearchAction.model_validate(x) for x in state['pending']]
            self.decisions=state['decisions'];self.stopped=state['stopped']
            self.stop_explanation=state['stop_explanation'];self.known_ids=set(state['known_ids'])
            self.error=state.get('error')

    def state(self):
        from pydantic import TypeAdapter
        return TypeAdapter(object).dump_python({'calls':self.session.calls,'full_calls':self.session.full_calls,
            'records':self.session.records,'log':self.session.log,'outputs':self.outputs,
            'pending':[a.model_dump() for a in self.pending],'decisions':self.decisions,
            'stopped':self.stopped,'stop_explanation':self.stop_explanation,
            'known_ids':sorted(self.known_ids),'error':self.error},mode='json')

    async def propose(self):
        from pydantic import TypeAdapter
        context=TypeAdapter(object).dump_json({'research_plan':self.plan.model_dump(mode='json'),
            'full_records':list(self.session.records.values()),'retrieval_responses':self.outputs,
            'operations_used':self.session.calls,'overall_cap':32}).decode()
        async with AsyncOpenAI(api_key=self.settings.openai_api_key,max_retries=0,
            default_headers={'Accept-Encoding':OPENAI_ACCEPT_ENCODING}) as client:
            agent=Agent(name='CheckpointResearcher',instructions=SELECTOR_INSTRUCTIONS,
                model=OpenAIResponsesModel(model=self.settings.openai_model,openai_client=client),
                model_settings=agent_model_settings(replace(self.settings,max_output_tokens=4000)),
                output_type=NextResearchActions,tools=[],handoffs=[])
            result=await Runner.run(agent,input=context,max_turns=1,run_config=RunConfig(tracing_disabled=True))
            observe_sdk_result(result)
        decision=NextResearchActions.model_validate(result.final_output.model_dump())
        if (decision.status=='continue') != bool(decision.actions):
            raise ValueError('Research decision status disagrees with its actions')
        self.decisions.append(decision.model_dump())
        self.pending=list(decision.actions)
        if decision.status=='no_useful_actions':
            self.stopped=True;self.stop_explanation=decision.stop_explanation

    def execute(self,action):
        arguments=action.arguments()
        if action.tool=='get_record' and arguments.get('record_id') not in self.known_ids:
            # Invalid requests consume allowance but cannot authorize unseen evidence.
            self.session.calls+=1
            response={'error':'undiscovered_record_id','instruction':'Search first; never guess IDs.'}
            self.session.event(tool=action.tool,arguments=arguments,status='invalid_input',record_ids=[],tool_calls=self.session.calls)
        elif action.tool=='search_records' and not arguments.get('query'):
            self.session.calls+=1;response={'error':'empty_query'}
            self.session.event(tool=action.tool,arguments=arguments,status='invalid_input',record_ids=[],tool_calls=self.session.calls)
        else:
            response=json.loads(self.session.invoke(action.tool,arguments))
        rows=response if isinstance(response,list) else [response]
        for row in rows:
            if not isinstance(row,dict):continue
            for field in ['record_id','source_record_id','related_record_id','supersedes_record_id']:
                if row.get(field):self.known_ids.add(row[field])
        self.outputs.append({'operation':self.session.calls,'tool':action.tool,'arguments':arguments,'result':response})
        return response

    async def advance(self,target,trace_tool=None,save_state=None):
        if target not in CHECKPOINTS:raise ValueError('Only 8, 16, 32 are predeclared')
        while self.session.calls<target and not self.stopped:
            if not self.pending:
                await self.propose()
                if save_state:save_state(self.state())
            if self.stopped:break
            action=self.pending.pop(0)
            if trace_tool:trace_tool(self,action)
            else:self.execute(action)
            if save_state:save_state(self.state())
        original=self.session.limits
        try:
            self.session.limits=original.model_copy(update={'max_tool_calls':target})
            pack=self.session.assemble(ResearchOutcome(stop_reason='no_matching_records' if not self.session.records
                else 'sufficient_for_question' if self.stopped else 'budget_exhausted'))
            if self.stopped:pack.missing_information.append('Researcher stopped early: '+self.stop_explanation)
            return pack
        finally:self.session.limits=original
