import sys
import pytest
from unittest.mock import patch
from main import main

@pytest.fixture
def mock_pipeline_functions():
    """Mocks the pipeline functions so we don't run heavy training during CLI routing tests."""
    with patch('main.prepare_data') as mock_prep, \
         patch('main.train_model') as mock_train, \
         patch('main.evaluate_model') as mock_eval, \
         patch('main.save_model') as mock_save, \
         patch('main.load_model') as mock_load, \
         patch('main.predict') as mock_pred:
        
        # Set up default dummy return values for the mocked functions
        mock_prep.return_value = (None, None, None, None)
        yield {
            "prepare": mock_prep,
            "train": mock_train,
            "evaluate": mock_eval,
            "save": mock_save,
            "load": mock_load,
            "predict": mock_pred
        }

def test_cli_prepare_flag(mock_pipeline_functions, capsys):
    # Simulate running: python main.py --prepare
    with patch.object(sys, 'argv', ['main.py', '--prepare']):
        main()
        
    # Check if the correct pipeline function was triggered
    mock_pipeline_functions["prepare"].assert_called_once()
    
    # Check if the expected terminal print statement occurred
    captured = capsys.readouterr()
    assert "data prepared" in captured.out

def test_cli_train_flag(mock_pipeline_functions, capsys):
    # Simulate running: python main.py --train
    with patch.object(sys, 'argv', ['main.py', '--train']):
        main()
        
    mock_pipeline_functions["prepare"].assert_called_once()
    mock_pipeline_functions["train"].assert_called_once()
    mock_pipeline_functions["save"].assert_called_once()
    
    captured = capsys.readouterr()
    assert "model trained and saved" in captured.out