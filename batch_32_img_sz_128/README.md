Data Set : RESIDE-6K

epoch : 100

steps_per_epoch : 750

val_steps_per_epoch : 100

batch : 8 * 4 -> gradient accumulation

image size : 128

widths : [64,128,256,256]

block_depth : 2

---

change plot_images()

learning_rate_schedular update( LateStartReduceLROnPlateau ) -> Starts updating to val_psnr from 74 epoch. 

best_cb -> monitor="val_psnr", mode="max"