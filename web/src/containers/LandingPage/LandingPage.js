import React, { Fragment } from 'react';
import PropTypes from 'prop-types';
import { connect } from "react-redux";
import LandingNavBar from "../../components/LandingNavBar/LandingNavBar"
import LandingProductHero from "../../components/LandingProductHero/LandingProductHero"
import LandingProductCategories from "../../components/LandingMainPage/LandingProductCategories"
import LandingTryRafiki from '../../components/LandingTryRafiki/LandingTryRafiki'
import LandingFooter from '../../components/LandingFooter/LandingFooter'


class LandingPage extends React.Component {
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
        <LandingProductHero />
        <LandingProductCategories />
        <LandingTryRafiki />
        <hr />
        <LandingFooter />
      </Fragment>
    )
  }
}

const mapStateToProps = state => ({
  auth: state.auth
})

export default connect(mapStateToProps)(LandingPage)