from common import BudgetType

class InvalidBudgetTypeException(Exception):
    pass

# Updates train job based on budget
def if_train_job_budget_reached(budget_type, budget_amount, completed_trials): 
    if budget_type == BudgetType.TRIAL_COUNT:
        max_trials = budget_amount 
        return len(completed_trials) >= max_trials
    else:
        raise InvalidBudgetTypeException()
