from unittest.mock import patch, MagicMock


def test_build_agent():
    """Test that build_agent constructs an AgentExecutor."""
    with patch("rag_agent.agent.get_llm") as mock_llm:
        mock_model = MagicMock()
        mock_model.bind_tools.return_value = mock_model
        mock_llm.return_value = mock_model

        from rag_agent.agent import build_agent
        executor = build_agent()

        assert executor is not None
        assert hasattr(executor, "invoke")
        assert len(executor.tools) == 3
