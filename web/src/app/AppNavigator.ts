import { createBrowserHistory } from 'history';

const history = createBrowserHistory();

const AppRoute = {
  LOGIN: '/login',
  DASHBOARD: '/',
  TRAIN_JOBS: '/train-jobs',
  TRAIN_JOB_DETAIL: '/train-jobs/:app/:appVersion',
  TRIAL_DETAIL: '/trial/:trialId',
  INFERENCE_JOBS: '/inference-jobs'
};

class AppNavigator {
  static AppRoute = AppRoute;

  goTo(route: string) {
    history.push(route);
  }
}

export default AppNavigator;
export { AppRoute, history };
