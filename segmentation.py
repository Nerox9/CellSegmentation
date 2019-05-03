from algorithm1 import Algorithm1
import cv2


def main():
    algorithm = Algorithm1()

    for i in range(10):
        image = algorithm.process(i, True)
        algorithm.test(image)

main()