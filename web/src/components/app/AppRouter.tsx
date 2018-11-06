import * as React from 'react';
import { Route, Switch, Router, Redirect } from 'react-router-dom';

import { history, AppRoute } from '../../app/AppNavigator';
import LoginPage from '../../pages/auth/LoginPage';
import DashboardLayout from '../../layouts/DashboardLayout';
import { AppUtils } from '../../App';

interface Props {
  appUtils: AppUtils;
}

class AppRouter extends React.Component<Props> {

  componentDidMount() {
    const { appUtils: { rafikiClient, appNavigator} } = this.props;

    const user = rafikiClient.getCurrentUser();
    if (!user) {
      appNavigator.goTo(AppRoute.LOGIN);
    }
  }

  render() {
    const { appUtils } = this.props;

    return (
      <Router history={history}>
        <Switch>
          <Route exact path={AppRoute.LOGIN} render={(props) => <LoginPage {...props} appUtils={appUtils} />}/>
          <Route path={AppRoute.DASHBOARD} render={(props) => <DashboardLayout {...props} appUtils={appUtils} />}/>
          <Redirect to={AppRoute.LOGIN} />
        </Switch>
      </Router>
    );
  }
}

export default AppRouter;