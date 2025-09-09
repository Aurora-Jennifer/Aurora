

def export_model_to_onnx(xgb_reg, feature_names: list[str], onnx_path: str, opset: int = 13):
    """
    Exports an XGBRegressor to ONNX (float32 input). Requires onnxmltools + skl2onnx.
    """
    from onnxmltools.convert import convert_xgboost
    from skl2onnx.common.data_types import FloatTensorType
    initial_type = [("input", FloatTensorType([None, len(feature_names)]))]
    onnx_model = convert_xgboost(xgb_reg, initial_types=initial_type, target_opset=opset)
    with open(onnx_path, "wb") as f:
        f.write(onnx_model.SerializeToString())
    return onnx_path
