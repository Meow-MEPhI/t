# graph_orchestrator.py
from typing import TypedDict
from langgraph.graph import StateGraph, START, END
from agent_bibliographer import BibliographerAgent
from agent_rubricator import RubricatorAgent
from agent_keyword import KeywordAgent
from agent_summariser import SummariserAgent
from rubricator_critic import CriticAgent
from agent_normal import NormalAgent
import time
from typing import Annotated, List, Literal
from IPython.display import Image, display
import operator



def should_continue_or_revise(state: dict) -> Literal["continue", "revise", "max_retries"]:
    """Решает, продолжать дальше или вернуться на переделку."""

    # Проверяем счетчик попыток
    revision_count = state.get("revision_count", 0)
    MAX_REVISIONS = 10  # Максимум 3 попытки

    if revision_count >= MAX_REVISIONS:
        return "max_retries"  # Превышен лимит, идём дальше с тем, что есть

    # Проверяем статус критика
    status_list = state.get("status", [])

    if "critic_rejected" in status_list:
        return "revise"  # Критик отклонил, возвращаемся к рубрикатору
    elif "critic_approved" in status_list:
        return "continue"  # Критик одобрил, идём дальше

    return "continue"  # По умолчанию продолжаем


def saferun(func, state: dict):
    while True:
        try:
            time.sleep(1)
            return func(state)
        except Exception as e:
            print(e)
            continue

# Определяем состояние графа
class GraphState(TypedDict):
    """Общее состояние для всех узлов графа."""
    article_url: str
    article_text: str
    rubric_result_keyword: str
    rubric_result_rubricator: str
    rubric_result_kritik: str
    rubric_result_normal: str
    rubric_result_summariser: str
    critique: str
    revision_count: int
    status: Annotated[List[str], operator.add]


def create_multi_agent_graph(auth_key: str):

    # Инициализируем агентов
    bibliographer = BibliographerAgent(auth_key=auth_key)
    rubricator = RubricatorAgent(auth_key=auth_key)
    keyword = KeywordAgent(auth_key=auth_key)
    normal = NormalAgent(auth_key=auth_key)
    summariser = SummariserAgent(auth_key=auth_key)
    critic_r = CriticAgent(auth_key=auth_key)


    # Создаем граф состояний
    workflow = StateGraph(GraphState)

    # Добавляем узлы (агентов) в граф
    workflow.add_node("bibliographer", lambda state: saferun(bibliographer.run, state))
    workflow.add_node("rubricator", lambda state: saferun(rubricator.run, state))
    workflow.add_node("critic_r", lambda state: saferun(critic_r.run, state))
    workflow.add_node("keyword", lambda state: saferun(keyword.run, state))
    workflow.add_node("normal", lambda state: saferun(normal.run, state))
    workflow.add_node("summariser", lambda state: saferun(summariser.run, state))

    # Определяем последовательность выполнения
    workflow.add_edge(START, "bibliographer")
    workflow.add_edge("bibliographer", "rubricator")
    workflow.add_edge("rubricator", "critic_r")

    workflow.add_conditional_edges(
        "critic_r",
        should_continue_or_revise,
        {
            "revise": "rubricator",  # Возврат на переделку
            "continue": "keyword",  # Переход к следующему агенту
            "max_retries": "keyword"  # Если превышен лимит, всё равно идём дальше
        }
    )

    workflow.add_edge("bibliographer", "keyword")
    workflow.add_edge("bibliographer", "normal")
    workflow.add_edge("bibliographer", "summariser")
    # workflow.add_edge("rubricator", "keyword")
    # workflow.add_edge("keyword", "normal")
    workflow.add_edge("normal", END)
    workflow.add_edge("keyword", END)
    workflow.add_edge("rubricator", END)
    workflow.add_edge("summariser", END)

    # Компилируем граф
    graph = workflow.compile()

    return graph



if __name__ == "__main__":

    AUTH_KEY = ""  # Вставьте ваш ключ GigaChat
    ARTICLE_URL = "https://habr.com/ru/companies/spbifmo/articles/343320/"

    graph = create_multi_agent_graph(AUTH_KEY)
    initial_state = {
        "article_url": ARTICLE_URL,
        "article_text": "",
        "rubric_result_rubricator": "",
        "rubric_result_keyword": "",
        "rubric_result_normal": "",
        "rubric_result_summariser": "",
        "revision_count": 0,
        "status": ["started"]
    }

    # Выполняем граф
    final_state = graph.invoke(initial_state)

    # Сохранить структуру в файл
    png_data = graph.get_graph().draw_mermaid_png()

    with open("graph_visualization.png", "wb") as f:
        f.write(png_data)


    print(final_state['rubric_result_rubricator'])
    print('\n')
    print(final_state['revision_count'])
