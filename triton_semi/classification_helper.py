# Copyright 2020-2022, NVIDIA CORPORATION & AFFILIATES. All rights reserved.
#
# Redistribution and use in source and binary forms, with or without
# modification, are permitted provided that the following conditions
# are met:
#  * Redistributions of source code must retain the above copyright
#    notice, this list of conditions and the following disclaimer.
#  * Redistributions in binary form must reproduce the above copyright
#    notice, this list of conditions and the following disclaimer in the
#    documentation and/or other materials provided with the distribution.
#  * Neither the name of NVIDIA CORPORATION nor the names of its
#    contributors may be used to endorse or promote products derived
#    from this software without specific prior written permission.
#
# THIS SOFTWARE IS PROVIDED BY THE COPYRIGHT HOLDERS ``AS IS'' AND ANY
# EXPRESS OR IMPLIED WARRANTIES, INCLUDING, BUT NOT LIMITED TO, THE
# IMPLIED WARRANTIES OF MERCHANTABILITY AND FITNESS FOR A PARTICULAR
# PURPOSE ARE DISCLAIMED.  IN NO EVENT SHALL THE COPYRIGHT OWNER OR
# CONTRIBUTORS BE LIABLE FOR ANY DIRECT, INDIRECT, INCIDENTAL, SPECIAL,
# EXEMPLARY, OR CONSEQUENTIAL DAMAGES (INCLUDING, BUT NOT LIMITED TO,
# PROCUREMENT OF SUBSTITUTE GOODS OR SERVICES; LOSS OF USE, DATA, OR
# PROFITS; OR BUSINESS INTERRUPTION) HOWEVER CAUSED AND ON ANY THEORY
# OF LIABILITY, WHETHER IN CONTRACT, STRICT LIABILITY, OR TORT
# (INCLUDING NEGLIGENCE OR OTHERWISE) ARISING IN ANY WAY OUT OF THE USE
# OF THIS SOFTWARE, EVEN IF ADVISED OF THE POSSIBILITY OF SUCH DAMAGE.

from PIL import Image
import numpy as np
import tritonclient.http as httpclient
from tritonclient.utils import triton_to_np_dtype
import json

class classification:
    
    #The number of classifications to be requested
    REQUEST_CLASSIFIACTION = 1 

    def __init__(self, model_name, url, port):
        self.model_name = model_name
        self.url = url
        self.port = port
        self.batch_size = 1
        self.connected = False
        self.prediction = []
        self.img_sim_class = []
        self.c = 0
        self.h = 0
        self.w = 0
        
    def parse_model(self, model_metadata, model_config):
        """
        Check the configuration of a model to make sure it meets the
        requirements for an image classification network (as expected by
        this client)
        """
        if len(model_metadata['inputs']) != 1:
            raise Exception("expecting 1 input, got {}".format(
                len(model_metadata['inputs'])))
        if len(model_metadata['outputs']) != 1:
            raise Exception("expecting 1 output, got {}".format(
                len(model_metadata['outputs'])))

        if len(model_config['input']) != 1:
            raise Exception(
                "expecting 1 input in model configuration, got {}".format(
                    len(model_config['input'])))

        self.input_metadata = model_metadata['inputs'][0]
        self.input_config = model_config['input'][0]
        self.output_metadata = model_metadata['outputs'][0]

        if self.output_metadata['datatype'] != "FP32":
            raise Exception("expecting output datatype to be FP32, model '" +
                            model_metadata['name'] + "' output type is " +
                            self._output_metadata['datatype'])

        # Output is expected to be a vector. But allow any number of
        # dimensions as long as all but 1 is size 1 (e.g. { 10 }, { 1, 10
        # }, { 10, 1, 1 } are all ok). Ignore the batch dimension if there
        # is one.
        output_batch_dim = (model_config['max_batch_size'] > 0)
        non_one_cnt = 0
        for dim in self.output_metadata['shape']:
            if output_batch_dim:
                output_batch_dim = False
            elif dim > 1:
                non_one_cnt += 1
                if non_one_cnt > 1:
                    raise Exception("expecting model output to be a vector")

        # Model input must have 3 dims, either CHW or HWC (not counting
        # the batch dimension)
        input_batch_dim = (model_config['max_batch_size'] > 0)
        expected_input_dims = 3 + (1 if input_batch_dim else 0)
        if len(self.input_metadata['shape']) != expected_input_dims:
            raise Exception(
                "expecting input to have {} dimensions, model '{}' input has {}".
                format(expected_input_dims, model_metadata['name'],
                        len(self.input_metadata['shape'])))

        #check the dim to choose format eg 3, 128, 128 = CHW
        max_pos = np.argmax(self.input_config['dims'])

        if(max_pos == 0):
            self.input_config['format'] = 'FORMAT_NHWC'
            self.h = self.input_metadata['shape'][1 if input_batch_dim else 0]
            self.w = self.input_metadata['shape'][2 if input_batch_dim else 1]
            self.c = self.input_metadata['shape'][3 if input_batch_dim else 2]
        else:
            self.input_config['format'] = 'FORMAT_NCHW'
            self.c = self.input_metadata['shape'][1 if input_batch_dim else 0]
            self.h = self.input_metadata['shape'][2 if input_batch_dim else 1]
            self.w = self.input_metadata['shape'][3 if input_batch_dim else 2]

        self.supports_batching = model_config['max_batch_size'] > 0


    def preprocess(self, img, format, dtype, c, h, w):
        """
        Pre-process an image to meet the size, type and format
        requirements specified by the parameters.
        """
        if c == 1:
            sample_img = img.convert('L') #8-bit pixels, black and white
        else:
            sample_img = img.convert('RGB') #3x8-bit pixels, true color


        resized_img = sample_img.resize((w, h), Image.Resampling.BILINEAR)
        resized = np.array(resized_img)
        if resized.ndim == 2:
            resized = resized[:, :, np.newaxis]

        npdtype = triton_to_np_dtype(dtype)
        scaled = resized.astype(npdtype) / 255

        # Swap to HWC if necessary
        if format == 'FORMAT_NCHW':
            ordered = np.transpose(scaled, (2, 0, 1))
        else:
            ordered = scaled

        # Channels are in RGB order. Currently model configuration data
        # doesn't provide any information as to other channel orderings
        # (like BGR) so we just assume RGB.
        return ordered
    
    def postprocess(self, responses, output_name, batch_size, supports_batching, output_filenames):
        """
        Post-process results to show classifications.
        """
        for file_name_batch, result_batch in zip(output_filenames, responses):
            
            output_array = result_batch.as_numpy(output_name)
            if supports_batching and len(output_array) != batch_size:
                raise Exception("expected {} results, got {}".format(
                    batch_size, len(output_array)))
            
            for output_filename, result_batch in zip(file_name_batch, output_array):
                # Include special handling for non-batching models
                if not supports_batching:
                    result = [result_batch]
                for  result in result_batch:
                    if output_array.dtype.type == np.object_:
                        cls = "".join(chr(x) for x in result).split(':')
                    else:
                        cls = result.split(':')
                    self.prediction.append(output_filename)
                    self.img_sim_class.append(cls[1])
          
    def request_generator(self, batched_image_data, input_name, output_name, dtype, classes):

        # Set the input data
        inputs = [httpclient.InferInput(input_name, batched_image_data.shape, dtype)]
        inputs[0].set_data_from_numpy(batched_image_data)

        outputs = [
          httpclient.InferRequestedOutput(output_name, class_count=classes)
        ]

        yield inputs, outputs
            
    def load_model(self):
        
        #connect to triton server
        url = '{}:{}'.format(self.url, self.port)
        self.triton_client = httpclient.InferenceServerClient(url=url, verbose=False)
        
        try:    
            self.triton_client.is_server_live()
            self.connected = True
        except:
            raise Exception('no conection to server. Check IP and Port')
            
        #load model
        model_metadata = self.triton_client.get_model_metadata(
                  model_name=self.model_name)

        model_config = self.triton_client.get_model_config(
                  model_name=self.model_name)

        self.parse_model(model_metadata, model_config)

    def model_ready(self, version_path, fs_pipe):
        '''
        arg:
            version_path: path with the last used model version (init version is 0)
            fs_pipe: filesystem of the pipeline (local, cloud...)
        
        return: true, if a higher version of the model is available since the loop befor.
        '''
        model_stat = self.triton_client.get_inference_statistics(self.model_name)
        
        stat_list = model_stat['model_stats']
        act_version = stat_list[0]['version']
        
        with fs_pipe.open(version_path, 'r') as fp:
                last_version = json.load(fp)
                 
        if int(act_version) > int(last_version['version']):
            with fs_pipe.open(version_path, 'w') as fp:
                
                new_model_version_dict = {'version': act_version}
                json.dump(new_model_version_dict, fp)
            return True
        else:
            return False
  
    def triton_predict(self, filenames):
      
        image_data = []
        for filename in filenames:
            img = Image.open(filename)
            image_data.append(
                self.preprocess(img, self.input_config['format'], self.input_metadata['datatype'], self.c, self.h, self.w))

        # Send requests of batch_size_of_request images. If the number of
        # images isn't an exact multiple of batch_size_of_request then just
        # start over with the first images until the batch is filled.
        responses = []
        output_filenames = []
        image_idx = 0
        last_request = False
        sent_count = 0

        #request to triton
        while not last_request:
            input_filenames = []
            repeated_image_data = []

            for _ in range(self.batch_size):
                input_filenames.append(filenames[image_idx])
                repeated_image_data.append(image_data[image_idx])
                image_idx = (image_idx + 1) % len(image_data)
                if image_idx == 0:
                    last_request = True

            if self.supports_batching:
                batched_image_data = np.stack(repeated_image_data, axis=0)
            else:
                batched_image_data = repeated_image_data[0]

            # Send request

            for inputs, outputs in self.request_generator(
                        batched_image_data, self.input_metadata['name'], self.output_metadata['name'], self.input_metadata['datatype'], self.REQUEST_CLASSIFIACTION):
                        sent_count += 1

                        responses.append(
                            self.triton_client.infer(self.model_name,
                                                inputs,
                                                request_id=str(sent_count),
                                                outputs=outputs))
                        
                        output_filenames.append(input_filenames)

        self.postprocess(responses, self.output_metadata['name'], self.batch_size, self.supports_batching,  output_filenames)
  