import abc
import tensorflow as tf
from enum import Enum
from configparser import ConfigParser


class NeuralNetwork(abc.ABC):
	@abc.abstractmethod
	def commit(self) -> None:
		pass

	@abc.abstractmethod
	def run(self, input_tensor: list) -> list:
		pass

	@abc.abstractmethod
	def train(self, training_data: list) -> None:
		pass

	@abc.abstractmethod
	def close(self) -> None:
		pass

	@abc.abstractmethod
	def save(self, path: str) -> None:
		pass

	@abc.abstractmethod
	def load(self) -> None:
		pass

	@abc.abstractclassmethod
	def restore(self, path: str):
		pass


class NeuralNetworkBase(NeuralNetwork):
	DEFAULT_SAVE_PATH = "./NeuralNetworks/saved"

	def __init__(self, input_shape: list, output_shape: list, name: str = "NNBase", gpu_enable: bool = True):
		"""
		:param input_shape: shape of allowed input tensors (equivalent to numpy's shape)
		:param output_shape: shape of the output tensor (equivalent to numpy's shape)
		:param name: used to differentiate between networks (should be set if the net is saved)
		:param gpu_enable: used to en-/disable gpu acceleration
		"""
		self.name = name
		self.input_shape = input_shape
		self.output_shape = output_shape

		self.input = tf.placeholder(dtype=tf.float32, shape=[None] + input_shape, name="x")
		self.learning_rate = tf.placeholder(dtype=tf.float32, shape=[])
		self.output = self.input		# the current output of the net. Once the net is committed it will be of shape self.output_shape
		self.session = None				# the session this net will be run in (can only be initialized once the net is build)
		self.saver = None				# saver object to allow saving of this net (can only be initialized once the net is build)
		self.optimizer = None			# node to be run in order to perform a training step
		self.merged_summary = None		# set of all summaries (can only be initialized once the net is build)

		self.net_config = ConfigParser()
		self.net_config["Format"] = {"input_shape": str(input_shape).rstrip("]").lstrip("["), "output_shape": str(output_shape).rstrip("]").lstrip("["), "n_layers": 0}
		self.net_config["Options"] = {}
		self.net_config["Stats"] = {"total_steps": 0, "total_time": 0}

		self.tf_config = tf.ConfigProto()
		self.tf_config.gpu_options.allow_growth = True
		self.tf_config.device_count["GPU"] = 1 if gpu_enable else 0

		self.committed = False

	def commit(self) -> None:
		pass

	def load(self) -> None:
		pass

	def run(self, input_tensor: list) -> list:
		pass

	def train(self, training_data: list) -> None:
		pass

	def close(self) -> None:
		pass

	def restore(self, path: str):
		pass

	def save(self, path: str = DEFAULT_SAVE_PATH) -> None:
		pass


class ActivationType(Enum):
	RELU = tf.nn.relu
	SIGMOID = tf.nn.sigmoid

	@staticmethod
	def get(act_type):
		if act_type == "RELU":
			return ActivationType.RELU
		if act_type == "SIGMOID":
			return ActivationType.SIGMOID

	@staticmethod
	def string_of(act_type):
		if act_type == ActivationType.RELU:
			return "RELU"
		if act_type == ActivationType.SIGMOID:
			return "SIGMOID"
