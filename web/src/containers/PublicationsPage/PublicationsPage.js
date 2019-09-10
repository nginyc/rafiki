import React, { Fragment } from 'react';
import PropTypes from 'prop-types';
import { connect } from "react-redux";
import LandingNavBar from "../../components/LandingNavBar/LandingNavBar"
import PubPage from "../../components/PublicationsPageComponents/PubPage"
import LandingFooter from '../../components/LandingFooter/LandingFooter'


class PublicationsPage extends React.Component {
  static propTypes = {
    auth: PropTypes.object
  }

  componentDidMount() {
    //  Scrolling to top of page when component loads
    window.scrollTo(0,0);
  }

  render() {
    const { auth } = this.props
    return (
      <Fragment>
        <LandingNavBar auth={auth} />
        <PubPage />
        <LandingFooter />
      </Fragment>
    )
  }
}

const mapStateToProps = state => ({
  auth: state.auth
})

export default connect(mapStateToProps)(PublicationsPage)