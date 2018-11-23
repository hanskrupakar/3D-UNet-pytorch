class CumulativeAverager:

    def __init__(self):
        self.count = 0
        self.sum = 0
        self.hist = []
    
    def update(self, newval):
        self.count += 1
        self.sum += newval
        self.hist.append(newval)
    
    def get_average(self):
        return self.sum/float(self.count)

    def get_full_history(self):
        return self.hist
