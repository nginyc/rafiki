from admin import Admin
from common import BudgetType

admin = Admin()

admin.create_train_job(
    app_name='fashion_mnist_app',
    budget_type=BudgetType.TRIAL_COUNT,
    budget_amount=10
)
