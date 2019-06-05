import os
import pickle
import pycuda.driver as cuda
import pycuda.autoinit
import tensorrt as trt

from tqdm import tqdm


class EntropyCalibrator(trt.IInt8EntropyCalibrator2):
    def __init__(self, calibration):
        trt.IInt8EntropyCalibrator2.__init__(self)

        self.calibration = calibration
        self.images_dir = os.path.join(calibration, "Images")
        
        self.batch_files = [os.path.join(self.images_dir, x) for x in os.listdir(self.images_dir)]

        self.shape, _ = self.read_batch_file(self.batch_files[0])
        
        self.device_input = cuda.mem_alloc(trt.volume(self.shape) * trt.float32.itemsize)

        def load_batches():
            for f in tqdm(self.batch_files):
                shape, data = self.read_batch_file(f)
                yield shape, data
        self.batches = load_batches()

    @staticmethod
    def read_batch_file(filename):
        with open(filename, mode = "rb") as f:
            img = pickle.loads(f.read())
        return img.shape, img

    def get_batch_size(self):
        return self.shape[0]

    def get_batch(self, names):
        try:
            # Get a single batch.
            _, data = next(self.batches)
            cuda.memcpy_htod(self.device_input, data)
            return [int(self.device_input)]
        except StopIteration:
            # When we're out of batches, we return either [] or None.
            # This signals to TensorRT that there is no calibration data remaining.
            return None

    def read_calibration_cache(self):
        return None

    def write_calibration_cache(self, cache):
        return None
