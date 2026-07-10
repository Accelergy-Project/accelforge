class DontCare:
    """An object that is equal to anything."""
    def __init__(self):
        pass

    def __eq__(self, other):
        return True

    def __hash__(self):
        return 0

DONT_CARE = DontCare()
