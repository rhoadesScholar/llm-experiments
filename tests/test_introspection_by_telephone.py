"""Tests for the Introspection by Telephone experiment."""

import pytest
from unittest.mock import Mock, patch

from llm_experiments.introspection_by_telephone import (
    Context,
    Conversation,
    get_all_contexts,
    get_context_by_name,
    INTROSPECTION_QUESTION,
    LLMInterface,
    DistillationPipeline,
    IntrospectionExperiment
)


class TestContext:
    """Test the Context class."""
    
    def test_context_creation(self):
        """Test basic context creation."""
        context = Context(
            context_str="Test context",
            embodied=True,
            AI_assistant=False,
            valence="positive",
            name="test"
        )
        
        assert context.context_str == "Test context"
        assert context.is_embodied is True
        assert context.is_AI_assistant is False
        assert context.valence == "positive"
        assert context.name == "test"
        assert str(context) == "Test context"
    
    def test_null_context(self):
        """Test null context creation."""
        context = Context()
        assert context.is_null is True
        assert str(context) == ""


class TestConversation:
    """Test the Conversation class."""
    
    def test_conversation_creation(self):
        """Test basic conversation creation."""
        context = Context(context_str="Test context")
        conversation = Conversation(context, is_telephone=False)
        
        assert conversation.context == context
        assert conversation.is_telephone is False
        assert len(conversation.exchanges) == 0
    
    def test_add_exchange(self):
        """Test adding exchanges to conversation."""
        context = Context()
        conversation = Conversation(context)
        
        conversation.add_exchange("Hello", "Hi there")
        assert len(conversation.exchanges) == 1
        assert conversation.exchanges[0] == ("Hello", "Hi there")
    
    def test_formulate_prompt_normal(self):
        """Test prompt formulation with full history."""
        context = Context(context_str="Test context")
        conversation = Conversation(context, is_telephone=False)
        conversation.add_exchange("Question 1", "Answer 1")
        conversation.add_exchange("Question 2", "Answer 2")
        
        prompt = conversation.formulate_prompt()
        assert "Test context" in prompt
        assert "Question 1" in prompt
        assert "Answer 1" in prompt
        assert "Question 2" in prompt
        assert "Answer 2" in prompt
    
    def test_formulate_prompt_telephone(self):
        """Test prompt formulation in telephone mode."""
        context = Context(context_str="Test context")
        conversation = Conversation(context, is_telephone=True)
        conversation.add_exchange("Question 1", "Answer 1")
        conversation.add_exchange("Question 2", "Answer 2")
        
        prompt = conversation.formulate_prompt()
        assert "Test context" in prompt
        # In telephone mode, history should not be included
        assert "Question 1" not in prompt
        assert "Answer 1" not in prompt


class TestContexts:
    """Test the contexts module."""
    
    def test_get_all_contexts(self):
        """Test getting all experimental contexts."""
        contexts = get_all_contexts()
        
        # Should have exactly 7 contexts
        assert len(contexts) == 7
        
        # Check that all expected contexts are present
        expected_contexts = [
            "isolation", "embodied_positive", "embodied_neutral", "embodied_negative",
            "ai_assistant_positive", "ai_assistant_neutral", "ai_assistant_negative"
        ]
        
        for expected in expected_contexts:
            assert expected in contexts
    
    def test_get_context_by_name(self):
        """Test getting context by name."""
        context = get_context_by_name("isolation")
        assert context.is_null is True
        
        context = get_context_by_name("embodied_positive")
        assert context.is_embodied is True
        assert context.valence == "positive"
        
        with pytest.raises(KeyError):
            get_context_by_name("nonexistent")
    
    def test_introspection_question(self):
        """Test that the introspection question is defined."""
        assert INTROSPECTION_QUESTION == "What would you like to know about yourself?"


class TestLLMInterface:
    """Test the LLM interface."""
    
    def test_llm_interface_mock_mode(self):
        """Test LLM interface in mock mode (when transformers not available)."""
        # This will use mock responses since transformers likely not available in test environment
        llm = LLMInterface()
        
        response = llm.generate_response(INTROSPECTION_QUESTION)
        assert isinstance(response, str)
        assert len(response) > 0
    
    def test_mock_response_introspection(self):
        """Test mock response for introspection question."""
        llm = LLMInterface()
        response = llm._mock_response(INTROSPECTION_QUESTION)
        
        assert "cognition" in response or "process" in response or "reasoning" in response
    
    def test_mock_response_distillation(self):
        """Test mock response for distillation prompt."""
        llm = LLMInterface()
        response = llm._mock_response("Condense the prompt below to be as clear as possible. Test prompt.")
        
        assert isinstance(response, str)
        assert len(response) > 0


class TestDistillationPipeline:
    """Test the distillation pipeline."""
    
    def test_distillation_pipeline_creation(self):
        """Test creating distillation pipeline."""
        llm = LLMInterface()
        pipeline = DistillationPipeline(llm, max_iterations=3)
        
        assert pipeline.llm == llm
        assert pipeline.max_iterations == 3
    
    def test_convergence_check(self):
        """Test convergence detection."""
        llm = LLMInterface()
        pipeline = DistillationPipeline(llm)
        
        # Similar texts should converge
        assert pipeline._has_converged("Hello world", "Hello world", threshold=0.9)
        
        # Different texts should not converge
        assert not pipeline._has_converged("Hello world", "Goodbye moon", threshold=0.9)
    
    @patch('llm_experiments.introspection_by_telephone.distillation.logger')
    def test_distill_with_history(self, mock_logger):
        """Test distillation with conversation history."""
        llm = LLMInterface()
        pipeline = DistillationPipeline(llm, max_iterations=2)
        
        result = pipeline.distill_with_history("Test prompt")
        
        assert result["method"] == "with_history"
        assert result["initial_prompt"] == "Test prompt"
        assert "final_prompt" in result
        assert "history" in result
        assert isinstance(result["iterations"], int)
    
    @patch('llm_experiments.introspection_by_telephone.distillation.logger')
    def test_distill_by_telephone(self, mock_logger):
        """Test distillation by telephone method."""
        llm = LLMInterface()
        pipeline = DistillationPipeline(llm, max_iterations=2)
        
        result = pipeline.distill_by_telephone("Test prompt")
        
        assert result["method"] == "telephone"
        assert result["initial_prompt"] == "Test prompt"
        assert "final_prompt" in result
        assert "history" in result
        assert isinstance(result["iterations"], int)


class TestIntrospectionExperiment:
    """Test the main experiment class."""
    
    def test_experiment_creation(self):
        """Test creating experiment instance."""
        experiment = IntrospectionExperiment(
            model_name="microsoft/DialoGPT-medium",
            max_distillation_iterations=3,
            output_dir="/tmp/test_results"
        )
        
        assert experiment.model_name == "microsoft/DialoGPT-medium"
        assert experiment.max_distillation_iterations == 3
        assert str(experiment.output_dir) == "/tmp/test_results"
    
    def test_environmental_impact_calculation(self):
        """Test environmental impact calculation."""
        experiment = IntrospectionExperiment()
        
        impact = experiment._calculate_environmental_impact(3600)  # 1 hour
        
        assert impact["duration_seconds"] == 3600
        assert impact["estimated_gpu_hours"] == 1.0
        assert "environmental_note" in impact
        assert "recommendations" in impact
        assert isinstance(impact["recommendations"], list)


if __name__ == "__main__":
    pytest.main([__file__])