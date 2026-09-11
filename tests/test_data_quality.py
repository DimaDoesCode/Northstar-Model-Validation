from validation.data_quality import validate_input_data

def test_target_and_id_are_present():
    result = validate_input_data("data/raw/application_train.csv")
    assert result["target_present"]
    assert result["id_present"]
    assert result["missing_target"] == 0
    assert result["target_values"] == [0, 1]
