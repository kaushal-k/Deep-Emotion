import tensorflow as tf
from stn import spatial_transformer_network as transformer

class Deep_Emotion(tf.keras.Model):
    def __init__(self):
        super(Deep_Emotion, self).__init__(name='')

        # feature extraction layers
        self.conv1 = tf.keras.layers.Conv2D(10, 3, kernel_initializer='random_normal')
        self.conv2 = tf.keras.layers.Conv2D(10, 3, kernel_initializer='random_normal')
        self.pool2 = tf.keras.layers.MaxPooling2D()

        self.conv3 = tf.keras.layers.Conv2D(10, 3, kernel_initializer='random_normal')
        self.conv4 = tf.keras.layers.Conv2D(10, 3, kernel_initializer='random_normal')
        self.pool4 = tf.keras.layers.MaxPooling2D()

        self.dropout4 = tf.keras.layers.Dropout(rate=0.5)

        # classification layers
        self.fc1 = tf.keras.layers.Dense(
            50,
            kernel_initializer='random_normal',
            kernel_regularizer=tf.keras.regularizers.L2(),
            bias_regularizer=tf.keras.regularizers.L2()
        )
        self.fc2 = tf.keras.layers.Dense(
            7,
            kernel_initializer='random_normal',
            kernel_regularizer=tf.keras.regularizers.L2(),
            bias_regularizer=tf.keras.regularizers.L2()
        )

        # spatial transformer network
        # padding and pool size so chosen such that output size is 90 as expected by loc_fc
        self.loc_net = tf.keras.Sequential([
            tf.keras.layers.Conv2D(8, 3, kernel_initializer='random_normal', padding='same'),
            tf.keras.layers.MaxPooling2D(4),
            tf.keras.layers.ReLU(),

            tf.keras.layers.Conv2D(10, 3, kernel_initializer='random_normal', padding='same'),
            tf.keras.layers.MaxPooling2D(4),
            tf.keras.layers.ReLU()
        ])

        # regression layer of localization net initialized to predict identity transform
        # output size=6 for affine transformations
        self.loc_fc = tf.keras.Sequential([
            tf.keras.layers.Dense(32, kernel_initializer='random_normal'),
            tf.keras.layers.ReLU(),
            tf.keras.layers.Dense(
                6,
                kernel_initializer='zeros',
                bias_initializer=tf.keras.initializers.Constant([1,0,0,0,1,0])
            )
        ])
    
    def stn(self, input):
        # localization - returns parameters for affine transformation
        out  = tf.reshape(self.loc_net(input), [-1,90])
        theta = tf.reshape(self.loc_fc(out), [-1,2,3])

        return theta
    
    def call(self, input, training=False):
        # feature extraction
        out = tf.nn.relu(self.conv1(input))
        out = self.conv2(out)
        out = tf.nn.relu(self.pool2(out))

        out = tf.nn.relu(self.conv3(out))
        out = self.conv4(out)
        out = tf.nn.relu(self.pool4(out))

        out = self.dropout4(out, training=training)

        # localization
        theta = tf.reshape(self.stn(input), [-1,2,3])

        # grid generation and sampling
        out = tf.reshape(transformer(out, theta), [-1,810])

        # classification
        out = tf.nn.relu(self.fc1(out))
        out = self.fc2(out)

        return out