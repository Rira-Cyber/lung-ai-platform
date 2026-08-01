class Threshold:
    @staticmethod
    def body(image, threshold=-600):
        return image > threshold

    @staticmethod
    def lung(image, threshold=-320):
        return image < threshold
