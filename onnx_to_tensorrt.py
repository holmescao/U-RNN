import tensorrt as trt

TRT_LOGGER = trt.Logger(trt.Logger.WARNING)

def build_engine(onnx_file_path, engine_file_path):
    """Takes an ONNX file and creates a TensorRT engine to run inference with"""
    with trt.Builder(TRT_LOGGER) as builder, builder.create_network(
        0 
    ) as network, builder.create_builder_config() as config, trt.OnnxParser(
        network, TRT_LOGGER
    ) as parser, trt.Runtime(
        TRT_LOGGER
    ) as runtime:
        config.set_memory_pool_limit(trt.MemoryPoolType.WORKSPACE, 10*(1 << 30)) # 10GB
        import os
        if not os.path.exists(onnx_file_path):
            print(
                "ONNX file {} not found, please run torch_to_onnx.py first to generate it.".format(onnx_file_path)
            )
            exit(0)
        print("Loading ONNX file from path {}...".format(onnx_file_path))
        with open(onnx_file_path, "rb") as model:
            print("Beginning ONNX file parsing")
            if not parser.parse(model.read()):
                print("ERROR: Failed to parse the ONNX file.")
                for error in range(parser.num_errors):
                    print(parser.get_error(error))
                return None

    # # 创建优化配置文件
    profile = builder.create_optimization_profile()
    # 添加对输入的优化配置
    min_input_shape = (1, 1,34, 500, 500)
    optimal_input_shape = (1, 1,34, 500, 500)
    max_input_shape = (1, 1,34, 500, 500)
    profile.set_shape("input_t", min_input_shape, optimal_input_shape, max_input_shape)

    # # 为 prev_encoder_state 和 prev_decoder_state 设置配置
    # encoder_shapes = [(1, 64, 500, 500), (1, 96, 250, 250), (1, 96, 125, 125)]
    # decoder_shapes = [(1, 96, 125, 125), (1, 96, 250, 250), (1, 64, 500, 500)]
    # state_names = ["prev_encoder_state_0", "prev_encoder_state_1", "prev_encoder_state_2",
    #                "prev_decoder_state_0", "prev_decoder_state_1", "prev_decoder_state_2"]
    # for name, shape in zip(state_names, encoder_shapes + decoder_shapes):
    #     profile.set_shape(name, shape, shape, shape)

    config.add_optimization_profile(profile)

    print("Completed parsing of ONNX file")
    print("Building an engine from file {}; this may take a while...".format(onnx_file_path))
    plan = builder.build_serialized_network(network, config)
    engine = runtime.deserialize_cuda_engine(plan)
    print("Completed creating Engine")
    with open(engine_file_path, "wb") as f:
        f.write(plan)
    return engine

# 调用函数
onnx_path = "net_model.onnx"
engine_path = "net_model.trt"

engine = build_engine(onnx_path, engine_path)
