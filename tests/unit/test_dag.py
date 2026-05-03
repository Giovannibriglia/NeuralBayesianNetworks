import pytest
import networkx as nx
from nbn.core.dag import DAG


def test_dag_basic():
    dag = DAG([("A", "B"), ("A", "C"), ("B", "D")])
    assert set(dag.nodes()) == {"A", "B", "C", "D"}
    assert ("A", "B") in dag.edges()
    assert dag.parents("B") == ["A"]
    assert dag.parents("A") == []
    assert "D" in dag.descendants("A")


def test_dag_topological_order():
    dag = DAG([("A", "B"), ("B", "C")])
    topo = dag.topological_order()
    assert topo.index("A") < topo.index("B") < topo.index("C")


def test_dag_rejects_cycle():
    with pytest.raises(ValueError, match="DAG"):
        DAG([("A", "B"), ("B", "A")])


def test_dag_extra_nodes():
    dag = DAG([("A", "B")], extra_nodes=["C"])
    assert "C" in dag.nodes()


def test_dag_from_networkx():
    g = nx.DiGraph()
    g.add_edge("X", "Y")
    dag = DAG(g)
    assert dag.parents("Y") == ["X"]


def test_dag_ancestors_descendants():
    dag = DAG([("A", "B"), ("B", "C"), ("C", "D")])
    assert "A" in dag.ancestors("D")
    assert "D" in dag.descendants("A")
    assert "C" not in dag.ancestors("A")
