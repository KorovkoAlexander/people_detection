import yaml
import os
import cv2
import numpy as np
import tensorrt as trt

from .common import allocate_buffers, do_inference
from .utils import PostProcessor, PriorBox, restore_bboxes


def load_cfg(cfg_name):
    cfg_path = os.path.join("./cfgs/", cfg_name + ".yaml")
    with open(cfg_path) as f:
        return yaml.load(f.read())


class PreprocessSSD:
    ssd_input_resolution = (800, 800)
    
    def __init__(self, ssd_input_resolution=None):
        if ssd_input_resolution is not None:
            assert isinstance(ssd_input_resolution, tuple)
            self.ssd_input_resolution = ssd_input_resolution

    def __call__(self, image_raw):
        image = cv2.resize(image_raw, self.ssd_input_resolution).astype(np.float32)
        image /= 255.0
        image = np.transpose(image, [2, 0, 1])[np.newaxis, :, :, :]
        # Convert the image to row-major order, also known as "C order":
        image = np.array(image, order='C')
        return image
    
    
class TRTDetector:
    def __init__(self, model_config, onnx_file_path=None, engine_file_path=None, use_preproc=True):
        if onnx_file_path is None and engine_file_path is None:
            raise ValueError("Please provide either ONNX or TRT file path!")
        self.cfg = model_config
        self.onnx_file_path = onnx_file_path
        self.engine_file_path = engine_file_path
        self.use_preproc = use_preproc
        
        self.TRT_LOGGER = trt.Logger()
        self.preprocessor = PreprocessSSD(tuple(self.cfg['IMAGE_SIZE']))
        
        prior_box = PriorBox(
            self.cfg['IMAGE_SIZE'],
            self.cfg['FEATURE_MAPS'],
            self.cfg['ASPECT_RATIOS'],
            self.cfg['SIZES'],
            archor_stride=self.cfg['STEPS']
        )
        
        self.priors = prior_box()
        self.post_processor = PostProcessor(self.cfg, self.priors)
        
        self.engine = self.get_engine()
        self.context = self.engine.create_execution_context()
        self.inputs, self.outputs, self.bindings, self.stream = allocate_buffers(self.engine)

    def build_engine(self):
        with trt.Builder(self.TRT_LOGGER) as builder, \
                builder.create_network() as network, \
                trt.OnnxParser(network, self.TRT_LOGGER) as parser:
            builder.max_workspace_size = 1 << 30 # 1GB
            builder.max_batch_size = 1
            # Parse model file
            if not os.path.exists(self.onnx_file_path):
                print(
                    'ONNX file {} not found, please run yolov3_to_onnx.py first to generate it.'.format(
                        self.onnx_file_path
                    )
                )
                exit(0)
            print(
                'Loading ONNX file from path {}...'.format(
                    self.onnx_file_path
                )
            )
            with open(self.onnx_file_path, 'rb') as model:
                print('Beginning ONNX file parsing')
                parser.parse(model.read())
            print('Completed parsing of ONNX file')
            print(
                'Building an engine from file {}; this may take a while...'.format(
                    self.onnx_file_path
                )
            )

            engine = builder.build_cuda_engine(network)
            print("Completed creating Engine")
            with open(self.engine_file_path, "wb") as f:
                f.write(engine.serialize())
            return engine
    
    def get_engine(self):
        if os.path.exists(self.engine_file_path):
            # If a serialized engine exists, use it instead of building an engine.
            print("Reading engine from file {}".format(self.engine_file_path))
            with open(self.engine_file_path, "rb") as f, \
                    trt.Runtime(self.TRT_LOGGER) as runtime:
                return runtime.deserialize_cuda_engine(f.read())
        else:
            return self.build_engine()
        
    def predict(self, image, threshold=0.6):
        image_preproc = image
        if self.use_preproc:
            image_preproc = self.preprocessor(image)
        
        assert image_preproc.shape == (1, 3, *self.cfg['IMAGE_SIZE']), f"{image_preproc.shape}"

        self.inputs[0].host = image_preproc
        trt_outputs = do_inference(
            self.context,
            bindings=self.bindings,
            inputs=self.inputs,
            outputs=self.outputs,
            stream=self.stream
        )

        out = (
            trt_outputs[0].reshape(1, -1, 4)[0][np.newaxis, :, :], # locs
            trt_outputs[1].reshape(1, -1, self.cfg['NUM_CLASSES'])[0][np.newaxis, :, :]  # confs
        )
        detections = self.post_processor(out)
        
        return restore_bboxes(detections, threshold, image.shape[1::-1])
