from train import train_until_threshold, fix_crash

size = (105, 145)
batch_size = 16
training_dir = "./images/item_links/training/"
test_dir = training_dir
training_name = "item_links"
new_training = True
epochs = 100
step_epoch = 100
expected_accuracy = 0.99

fix_crash()
train_until_threshold(training_name, training_dir, size, test_dir,
                      expected_accuracy, batch_size, epochs, step_epoch, new_training)
