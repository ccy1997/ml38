import pandas as pd
from sklearn.preprocessing import MinMaxScaler
import tensorflow as tf


df = pd.read_csv('housing.csv')  

df_train = df[:14449]    # 70% training data 
df_test = df[14449:]      # 30% test data

scaler = MinMaxScaler()

# WE NEED TO NORMALISE OUR CODE HERE RATHER THAN IN THE INPUT DATA SO THAT WE CAN DENORMALISE IT LATER IN THE PROGRAM 
X_train = scaler.fit_transform(df_train.drop(['median_house_value'],axis=1).as_matrix())
y_train = scaler.fit_transform(df_train['median_house_value'].as_matrix())

X_test = scaler.fit_transform(df_test.drop(['median_house_value'],axis=1).as_matrix())
y_test = scaler.fit_transform(df_test['median_house_value'].as_matrix())

# NOT SURE IF WE SHOULD BE NORMALISING THE DATA HERE AND THEN DENORMALISING IT RATHER THAN NORMALISING IT BEFORE HAND
def denormalise(df,normalised_data):
    df = df['median_house_value'].values.reshape(-1,1)
    normalised_data = normalised_data.reshape(-1,1)
    scale = MinMaxScaler()
    original = scale.fit_transform(df)
    re_scaled = original.inverse_transform(normalised_data)
 

l1_nodes = 12
l2_nodes = 8

# THIS MAY CHANGE AFTER WE DO FEATURE SELECTION
num_features = 10

def neural_network_model(housing_data, num_features):
    W_1 = tf.Variable(tf.random_normal([num_features, l1_nodes]))
    b_1 = tf.Variable(tf.random_normal([l1_nodes]))
    layer_1 = tf.add(tf.matmul(housing_data,W_1), b_1)
    layer_1 = tf.nn.relu(layer_1)


    W_2 = tf.Variable(tf.random_normal([l1_nodes,l2_nodes]))
    b_2 = tf.Variable(tf.random_normal([l2_nodes]))
    layer_2 = tf.add(tf.matmul(layer_1,W_2), b_2)
    layer_2 = tf.nn.relu(layer_2)


    W_output = tf.Variable(tf.random_normal([l2_nodes,1]))
    b_output = tf.Variable(tf.zeros([1]))
    output = tf.add(tf.matmul(layer_2,W_output), b_output)

    return output


xs = tf.placeholder("float")
ys = tf.placeholder("float")

output = neural_network_model(xs,num_features)

loss = tf.reduce_mean(tf.square(output-ys))

train = tf.train.GradientDescentOptimizer(0.005).minimize(loss)

num_iterations = 20

training_loss = []
test_loss = []

with tf.Session() as sess:

    saver = tf.train.Saver()
    sess.run(tf.global_variables_initializer())
    
    for iteration in range(num_iterations):
        for i in range(X_train.shape[0]):
            sess.run([loss,train],feed_dict = {xs:X_train[i,:].reshape(1,3), ys:y_train[i]})
          

        training_loss.append(sess.run(loss, feed_dict={xs:X_train,ys:y_train}))
        test_loss.append(sess.run(loss, feed_dict={xs:X_test,ys:y_test}))
        print('Epoch :',i,'Training loss :',training_loss[i])

    prediction = sess.run(output, feed_dict={xs:X_test})

    print('Loss :',sess.run(loss, feed_dict={xs:X_test,ys:y_test}))
    y_test = denormalise(df_test,y_test)
    prediction = denormalise(df_test,prediction)

    print('Loss :',sess.run(loss, feed_dict={xs:X_test,ys:y_test}))
  

    if input('Save model ? [Y/N]') == 'Y':
        saver.save(sess,'neural_net_model')
        print('saved')