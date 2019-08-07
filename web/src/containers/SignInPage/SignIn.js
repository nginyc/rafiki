import React from 'react';
import PropTypes from 'prop-types';
import { withStyles } from '@material-ui/core/styles';
import { Field, Form, FormSpy } from 'react-final-form';
import Typography from '../../components/LandingComponents/Typography';
import AppFooter from '../../components/LandingFooter/LandingFooter';
import AppAppBar from '../../components/LandingNavBar/LandingNavBar';
import AppForm from '../../components/LandingAppForm/LandingAppForm';
import { email, required } from '../../components/LandingAppForm/validation';
import RFTextField from '../../components/LandingAppForm/RFTextField';
import FormButton from '../../components/LandingAppForm/FormButton';
import FormFeedback from '../../components/LandingAppForm/FormFeedback';

import { compose } from "redux"
import { connect } from "react-redux"
import * as actions from "../Root/actions"
import { Redirect } from "react-router-dom"


const styles = theme => ({
  form: {
    marginTop: theme.spacing(1) * 6,
  },
  button: {
    marginTop: theme.spacing(1) * 3,
    marginBottom: theme.spacing(1) * 2,
  },
  feedback: {
    marginTop: theme.spacing(1) * 2,
  },
});

class SignIn extends React.Component {
  state = {
    sent: false,
  };

  static propTypes = {
    classes: PropTypes.object,
  };

  validate = values => {
    const errors = required(['email', 'password'], values, this.props);

    if (!errors.email) {
      const emailError = email(values.email, values, this.props);
      if (emailError) {
        errors.email = email(values.email, values, this.props);
      }
    }

    return errors;
  };

  handleSubmit = (values) => {
    console.log(values)
    const authData = Object.assign(
      {},
      {
        email: values.email,
        password: values.password
      }
    )
    this.props.signInRequest(authData)
  };

  render() {
    const {
      classes,
      authError,
      authStatus
    } = this.props;

    //const { sent } = this.state;

    if (authStatus) {
      return <Redirect to="/console/datasets/list-dataset" />
    }

    return (
      <React.Fragment>
        <AppAppBar />
        <AppForm>
          <React.Fragment>
            <Typography variant="h3" gutterBottom marked="center" align="center">
              Sign In
            </Typography>
            <Typography variant="body2" align="center">
              {'Not a user yet? '}
                Contact the administrator to create an account for you first.
            </Typography>
          </React.Fragment>
          {authError && "Log in Error " + authError}
          <Form
            onSubmit={this.handleSubmit}
            subscription={{
              submitting: true,
              valid: true
            }}
            validate={this.validate}
          >
            {({ handleSubmit, submitting, valid }) => (
              <form onSubmit={handleSubmit} className={classes.form} noValidate>
                <Field
                  autoComplete="email"
                  autoFocus
                  component={RFTextField}
                  disabled={submitting}
                  fullWidth
                  label="Email"
                  margin="normal"
                  name="email"
                  required
                  size="large"
                />
                <Field
                  fullWidth
                  size="large"
                  component={RFTextField}
                  disabled={submitting}
                  required
                  name="password"
                  autoComplete="current-password"
                  label="Password"
                  type="password"
                  margin="normal"
                />
                <FormSpy subscription={{ submitError: true }}>
                  {({ submitError }) =>
                    submitError ? (
                      <FormFeedback className={classes.feedback} error>
                        {submitError}
                      </FormFeedback>
                    ) : null
                  }
                </FormSpy>
                <FormButton
                  className={classes.button}
                  disabled={submitting || !valid}
                  size="large"
                  color="secondary"
                  fullWidth
                >
                  {submitting ? 'In progress…' : 'Sign In'}
                </FormButton>
              </form>
            )}
          </Form>
        </AppForm>
        <AppFooter />
      </React.Fragment>
    );
  }
}


const mapStateToProps = state => ({
  authError: state.Root.error,
  authStatus: !!state.Root.token,
})

const mapDispatchToProps = {
  signInRequest: actions.signInRequest
}


export default compose(
  connect(
    mapStateToProps,
    mapDispatchToProps
  ),
  withStyles(styles)
)(SignIn);
