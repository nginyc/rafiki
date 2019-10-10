import React from 'react';
import { withStyles } from '@material-ui/core/styles';

import LayoutBody from '../LandingComponents/LayoutBody';
import Typography from '../LandingComponents/Typography';

import '../LandingMainPage/Overview.css'


const styles = theme => ({
  root: {
    marginTop: theme.spacing(1) * 8,
  },
})


function ContactComponents(props) {
  const { classes } = props;
  return (
    <React.Fragment>
      <LayoutBody className={classes.root} component="section" width="large">
        <Typography variant="h3" gutterBottom marked="center" align="center">
          Contact
        </Typography>
        <div className="section_center">
          <Typography variant="h5" gutterBottom marked="center" align="left">
            <b>DBsystem NUS School of Computing</b>
          </Typography>
          <div className="center__description">
            <p className="description__text">
              <b>Address:</b><br />
              School of Computing, COM 1 Building,
13 Computing Drive, National University of Singapore<br />
              Singapore, 117417<br />
              <b>Website: </b><br />
              <a href="https://www.comp.nus.edu.sg/~dbsystem/" target="_blank" rel="noopener noreferrer">https://www.comp.nus.edu.sg/~dbsystem/</a><br />
            </p>
          </div>

            <Typography variant="h5" gutterBottom marked="center" align="left">
              <b>Find us on GitHub</b>
            </Typography>
            <div className="center__description">
              <p className="description__text">
                <b>Repository: </b><br />
                <a href="https://github.com/nginyc/rafiki" target="_blank" rel="noopener noreferrer">https://github.com/nginyc/rafiki</a>
              </p>
            </div>
          </div>
      </LayoutBody>
    </React.Fragment>
      );
    }
    
    export default withStyles(styles)(ContactComponents);
