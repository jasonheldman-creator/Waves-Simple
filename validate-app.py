# Updated CI Validation Rules

class ValidateApp:
    def __init__(self):
        # Define the allowed tabs
        self.allowed_tabs = ["Overview", "Alpha Attribution", "Adaptive Intelligence", "Governance & Operations", "Audit Trail", "Glossary & Concepts"]
        self.optional_tabs = ["decommission/tab", "placeholder state"]

    def validate_tabs(self, tabs):
        for tab in tabs:
            if tab in self.allowed_tabs:
                print(f"Tab '{tab}' is validated.")
            elif tab in self.optional_tabs:
                print(f"Tab '{tab}' is optional and can be skipped.")
            else:
                print(f"Tab '{tab}' is not recognized and should be reviewed.")

# Example usage
validator = ValidateApp()
validator.validate_tabs(["Overview", "decommission/tab", "Glossary & Concepts", "unknown tab"])
