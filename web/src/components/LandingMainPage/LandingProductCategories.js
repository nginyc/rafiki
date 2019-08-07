import React from 'react';
import PropTypes from 'prop-types';
import { withStyles } from '@material-ui/core/styles';

import LayoutBody from '../LandingComponents/LayoutBody';
import Typography from '../LandingComponents/Typography';

import './Overview.css'
import rafikiArch from "../../assets/rafikiArch.png"


const styles = theme => ({
  root: {
    marginTop: theme.spacing(1) * 8,
    marginBottom: theme.spacing(1) * 4,
  },
});

function ProductCategories(props) {
  const { classes } = props;

  return (
    <React.Fragment>
      <LayoutBody className={classes.root} component="section" width="large">
        <Typography variant="h4" marked="center" align="center" component="h2">
          Architecture
        </Typography>
        <div className="section_center">
          <div className="center__description">
            <p className="description__text">
            Rafiki’s system architecture consists of 3 static components, 2 central databases, 3 types of dynamic components, and 1 client-side SDK, which can be illustrated with a 3-layer architecture diagram.
            </p>
          </div>
          <div className="img-container">
            <img src={rafikiArch} alt="rafikiArch" className="fullWidthImg" />
          </div>
        </div>
      </LayoutBody>
    </React.Fragment>
  );
}

ProductCategories.propTypes = {
  classes: PropTypes.object.isRequired,
};

export default withStyles(styles)(ProductCategories);
