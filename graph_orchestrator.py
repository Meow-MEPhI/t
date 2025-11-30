# graph_orchestrator.py
from typing import TypedDict
from langgraph.graph import StateGraph, START, END
from agent_bibliographer import BibliographerAgent
from agent_rubricator import RubricatorAgent
from agent_keyword import KeywordAgent
from agent_kritik import KritikAgent
from agent_normal import NormalAgent
from IPython.display import Image, display

# Определяем состояние графа
class GraphState(TypedDict):
    """Общее состояние для всех узлов графа."""
    article_url: str
    article_text: str
    rubric_result_keyword: str
    rubric_result_rubricator: str
    rubric_result_kritik: str
    rubric_result_normal: str
    status: str


def create_multi_agent_graph(auth_key: str):

    # Инициализируем агентов
    bibliographer = BibliographerAgent(auth_key=auth_key)
    rubricator = RubricatorAgent(auth_key=auth_key)
    keyword = KeywordAgent(auth_key=auth_key)
    normal = NormalAgent(auth_key=auth_key)

    # Создаем граф состояний
    workflow = StateGraph(GraphState)

    # Добавляем узлы (агентов) в граф
    workflow.add_node("bibliographer", bibliographer.run)
    workflow.add_node("rubricator", rubricator.run)
    workflow.add_node("keyword", keyword.run)
    workflow.add_node("normal", normal.run)

    # Определяем последовательность выполнения
    workflow.add_edge(START, "bibliographer")
    workflow.add_edge("bibliographer", "rubricator")
    workflow.add_edge("rubricator", "keyword")
    workflow.add_edge("keyword", "normal")
    workflow.add_edge("normal", END)

    # Компилируем граф
    graph = workflow.compile()

    return graph



if __name__ == "__main__":

    AUTH_KEY = ""  # Вставьте ваш ключ GigaChat
    ARTICLE_URL = ""

    graph = create_multi_agent_graph(AUTH_KEY)
    initial_state = {
        "article_url": ARTICLE_URL,
        "article_text": "",
        "rubric_result_rubricator": "",
        "rubric_result_keyword": "",
        "rubric_result_normal": "",
        "status": "started"
    }

    # Выполняем граф
    final_state = graph.invoke(initial_state)

    # Сохранить структуру в файл
    png_data = graph.get_graph().draw_mermaid_png()

    with open("graph_visualization.png", "wb") as f:
        f.write(png_data)


    print(final_state['rubric_result_rubricator'])
    print('\n')
    print(final_state['rubric_result_keyword'])
    print('\n')
    print(final_state['rubric_result_normal'])
