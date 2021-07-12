from train import train_until_threshold, fix_crash

size = (46, 46)
batch_size = 16
epochs = 100
training_dir = "./images/currency_types/training/"
test_dir = training_dir
training_name = "currency_type"
new_training = True
epochs = 120
step_epoch = 100
expected_accuracy = 1.0

fix_crash()
train_until_threshold(training_name, training_dir, size, test_dir,
                      expected_accuracy, batch_size, epochs, step_epoch, new_training)
