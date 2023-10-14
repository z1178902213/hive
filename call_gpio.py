from periphery import GPIO
import time

# Open GPIO /dev/gpiochip0 line 10 with input direction
gpio_in = GPIO(89, "in")

# Open GPIO /dev/gpiochip0 line 12 with output direction
gpio_out = GPIO(81, "out")

start = time.time()

count = 0
out = True

while True:
    if time.time() - start > 2:
        print(f"输出{out}，改变状态为{not out}")
        gpio_out.write(out)
        out = not out
        start = time.time()
        count += 1
    if count >= 1000:
        break

gpio_in.close()
gpio_out.close()
