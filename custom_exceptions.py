class RequiredFieldsNotFoundError(Exception):
    def __init__(self):
        self.message = f"Make sure that the data set contains 'sentence' and 'category' fields!"
        super().__init__(self.message)
