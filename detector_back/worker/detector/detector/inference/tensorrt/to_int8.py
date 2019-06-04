import random
import tensorrt as trt
import pycuda.driver as cuda
import pycuda.autoinit

import calibrator
import common

import click

TRT_LOGGER = trt.Logger()


def build_int8_engine(onnx_file_path, calib):
    with trt.Builder(TRT_LOGGER) as builder, \
            builder.create_network() as network, \
            trt.OnnxParser(network, TRT_LOGGER) as parser:
        builder.max_batch_size = calib.get_batch_size()
        builder.max_workspace_size = common.GiB(1)
        builder.int8_mode = True
        builder.int8_calibrator = calib
        
        with open(onnx_file_path, 'rb') as model:
            parser.parse(model.read())
        engine = builder.build_cuda_engine(network)
    return engine


def load_random_batch(calib):
    batch = random.choice(calib.batch_files)
    _, data = calib.read_batch_file(batch)
    return data


@click.command()
@click.option("--dataset", default = "/home/a.korovko/Code/Datasets/Calibration")
@click.option("--onnx_file", default = "../ssd_resnet18_train_mix_800.onnx")
@click.option("--save_name", default = "ssd_resnet18_train_min_int.trt")
def main(dataset, onnx_file, save_name):
    calib = calibrator.EntropyCalibrator(dataset)
    # We will use the calibrator batch size across the board.
    # This is not a requirement, but in this case it is convenient.
    batch_size = calib.get_batch_size()

    with build_int8_engine(onnx_file, calib) as engine, \
            engine.create_execution_context() as context:
        # Allocate engine buffers.
        inputs, outputs, bindings, stream = common.allocate_buffers(engine)

        inputs[0].host = load_random_batch(calib)
        output = common.do_inference(
            context,
            bindings=bindings,
            inputs=inputs,
            outputs=outputs,
            stream=stream,
            batch_size=batch_size
        )
        
        with open(save_name, "wb") as f:
                f.write(engine.serialize())
    print("Done!")


if __name__ == '__main__':
    main()
