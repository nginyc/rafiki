import PropTypes from "prop-types";
import React, { Fragment } from "react";
import * as actions from "./actions"
import { connect } from "react-redux";

import NotificationArea from "../../components/RootComponents/NotificationArea";

class Root extends React.PureComponent {
  static propTypes = {
    children: PropTypes.oneOfType([
      PropTypes.arrayOf(PropTypes.node),
      PropTypes.node
    ]),
    isAuthenticated: PropTypes.bool.isRequired,
    notification: PropTypes.object,
    handleNotificationClose: PropTypes.func,
    onTryAutoSignup: PropTypes.func,
  };

  componentDidMount() {
    this.props.onTryAutoSignup();
    console.log("HAA", this.props.isAuthenticated)
  }

  render() {
    const {
      children,
      notification,
      handleNotificationClose
    } = this.props;

    return (
      <Fragment>
        <NotificationArea
          handleClose={handleNotificationClose}
          message={notification.message}
          open={notification.show}
        />
        {children}
      </Fragment>
    );
  }
}


const mapStateToProps = state => ({
  isAuthenticated: state.Root.token !== null,
  notification: state.Root.notification
});

const mapDispatchToProps = {
  handleNotificationClose: actions.notificationHide,
  onTryAutoSignup: actions.authCheckState
}


export default connect(
  mapStateToProps,
  mapDispatchToProps
)(Root)
