class Dashboard:
    def __init__(self):
        self.data = {}

    def update_data(self, new_data):
        self.data.update(new_data)

    def get_data(self):
        return self.data
