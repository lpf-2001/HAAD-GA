import tensorflow as tf

from Save_model.AWF import AWFNet
from Save_model.DF import DFNet
from Save_model.VarCNN import VarCNN


class ModelWrapper(tf.keras.Model):
    def __init__(self, modelname, datasetname, is_argmax=False, input_format='BCF'):  # BFC or BCF
        super(ModelWrapper, self).__init__()
        self.is_argmax = is_argmax
        self.input_format = input_format
        self.model = self.build_model(modelname, datasetname)

    def build_model(self, modelname, datasetname):
        LENGTH = 5000

        if datasetname == "DF":
            NB_CLASSES = 95
        elif datasetname == "AWF":
            NB_CLASSES = 103
        else:
            raise ValueError("Unsupported dataset name")

        if modelname == 'AWF':
            INPUT_SHAPE = (LENGTH, 1)
            model = AWFNet.build(input_shape=INPUT_SHAPE, classes=NB_CLASSES)
            model.load_weights(
                f'./Save_model/Save_Classfymodel/{datasetname}_DataSet/{modelname}/AWF_model_in_{datasetname}_DataSet.h5')

        elif modelname == 'DF':
            INPUT_SHAPE = (LENGTH, 1)
            model = DFNet.build(input_shape=INPUT_SHAPE, classes=NB_CLASSES)
            model.load_weights(
                f'./Save_model/Save_Classfymodel/{datasetname}_DataSet/{modelname}/DF_model_in_{datasetname}_DataSet.h5')

        elif modelname == 'VarCNN':
            INPUT_SHAPE = (LENGTH, 1)
            model = VarCNN.build(input_shape=INPUT_SHAPE, classes=NB_CLASSES)
            model.load_weights(
                f'./Save_model/Save_Classfymodel/{datasetname}_DataSet/{modelname}/VarCNN_model_in_{datasetname}_DataSet.h5')


        else:
            raise ValueError("Unsupported model name")

        print(f'success load {model.name} in {datasetname} dataset')
        return model

    # def call(self, inputs):
    #     """
    #     这里做输入输出shape的适配
    #     输入shape:[Batch,C,Feature]
    #     输出shape:[Batch,Classfi]
    #
    #     """
    #     if self.model.name=='AWF':
    #         inputs =tf.transpose(inputs, perm=[0, 2, 1])
    #         output = self.model(inputs)
    #         if self.is_argmax:
    #             output = tf.argmax(output, axis=1)
    #     elif self.model.name=='DF':
    #         inputs = tf.transpose(inputs, perm=[0, 2, 1])
    #         output = self.model(inputs)
    #         if self.is_argmax:
    #             output = tf.argmax(output, axis=1)
    #     else:
    #         print("ModelWrapper Error")
    #         exit()
    #
    #     return output

    def call(self, inputs):
        """
        这里做输入输出shape的适配
        输入shape:[Batch,C,Feature]
        输出shape:[Batch,Classfi]

        """
        if self.model.name=='AWF':
            if self.input_format == 'BCF':
                inputs =tf.transpose(inputs, perm=[0, 2, 1])
            output = self.model(inputs)
            if self.is_argmax:
                output = tf.argmax(output, axis=1)
        elif self.model.name=='DF':
            if self.input_format == 'BCF':
                inputs = tf.transpose(inputs, perm=[0, 2, 1])
            output = self.model(inputs)
            if self.is_argmax:
                output = tf.argmax(output, axis=1)
        elif self.model.name=='VarCNN':
            if self.input_format == 'BCF':
                inputs = tf.transpose(inputs, perm=[0, 2, 1])
            output = self.model(inputs)
            if self.is_argmax:
                output = tf.argmax(output, axis=1)
        else:
            print("ModelWrapper Error")
            exit()

        return output


    def predict(self, inputs):
        return self.call(inputs)
