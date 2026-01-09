"""
BiLSTM + Multi-Head Attention model for video summarization
"""

import tensorflow as tf
from tensorflow.keras import layers, models


class BiLSTMAttentionModel(models.Model):
    def __init__(self, feature_dim, num_heads=4, lstm_units=64, **kwargs):
        super(BiLSTMAttentionModel, self).__init__(**kwargs)

        self.feature_dim = feature_dim
        self.num_heads = num_heads
        self.lstm_units = lstm_units

        self.bilstm = layers.Bidirectional(
            layers.LSTM(lstm_units, return_sequences=True)
        )

        self.attention = layers.MultiHeadAttention(
            num_heads=num_heads,
            key_dim=feature_dim // num_heads
        )

        self.reconstruction_layer = layers.TimeDistributed(
            layers.Dense(feature_dim, activation="linear")
        )

    def call(self, inputs):
        x = self.bilstm(inputs)
        x = self.attention(x, x)
        return self.reconstruction_layer(x)

    def get_config(self):
        config = super().get_config()
        config.update({
            "feature_dim": self.feature_dim,
            "num_heads": self.num_heads,
            "lstm_units": self.lstm_units
        })
        return config

    @classmethod
    def from_config(cls, config):
        return cls(**config)
