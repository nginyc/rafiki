import { createBrowserHistory } from 'history';

const history = createBrowserHistory();

const AppRoute = {
  LOGIN: '/login',
  DASHBOARD: '/admin',
  DATASETS: '/admin/datasets',
  TRAIN_JOBS: '/admin/jobs',
  TRAIN_JOB_DETAIL: '/admin/jobs/trials/:app/:appVersion',
  TRIAL_DETAIL: '/admin/jobs/trials/details/:trialId',
  INFERENCE_JOBS: '/admin/inference-jobs' 
};

class AppNavigator {
  static AppRoute = AppRoute;

  goTo(route: string) {
    history.push(route);
  }
}

export default AppNavigator;
export { AppRoute, history };
