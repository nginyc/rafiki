import React from 'react';
import { withStyles } from '@material-ui/core/styles';

import Typography from '../LandingComponents/Typography';

import '../LandingMainPage/Overview.css'
import VisualizeDataEvolution from "../../assets/VisualizeDataEvolution.png"
import DiffQuery from "../../assets/DiffQuery.png"
import DataDedup from "../../assets/DataDedup.png"


const styles = theme => ({
  root: {
    marginTop: theme.spacing(1) * 8,
  },
})


function DemoFeaturesComponents(props) {
  const { classes } = props;
  return (
    <div className={classes.root}>
      <Typography variant="h3" gutterBottom marked="center" align="center">
        Demonstration
      </Typography>
      <div className="section_copy">
        <div className="conversation__description">
          <div className="description__heading">
            Visualize data evolution
          </div>
          <p className="description__text">
<b>Data versions </b>in ForkBase are generated according to the Merkle root hash
of data chunks, and encoded using the RFC 4648
Base32 alphabet. Such versioning
scheme enables ForkBase to provide <b>tamper evidence </b>against
malicious storage providers.
          </p>
        </div>
        <div className="img-container">
          <img src={VisualizeDataEvolution} alt="VisDataEvo" className="landing-img" />
        </div>
      </div>

      <div className="section_copy">
        <div className="conversation__description">
          <div className="description__heading">
            Differential query
          </div>
          <p className="description__text">
ForkBase supports fast differential query between data versions. <b>Data
differences are highlighted </b>at multiple scopes from
dataset to data entry. This resembles the Git-diff
utility to help user identify the changes of data.
          </p>
        </div>
        <div className="img-container">
          <img src={DiffQuery} alt="DiffQuery" className="landing-img" />
        </div>
      </div>

      <div className="section_copy">
        <div className="conversation__description">
          <div className="description__heading">
            Data deduplication
          </div>
          <p className="description__text">
When two datasets share a large portion of duplicated data, ForkBase can effectively
detect such redundancy and consequently<b> store only the marginal difference </b>into the underlying immutable storage for
the second loading.
          </p>
        </div>
        <div className="img-container">
          <img
            src={DataDedup}
            alt="DataDedup"
            className="landing-img"
          />
        </div>
      </div>

    </div>
  );
}

export default withStyles(styles)(DemoFeaturesComponents);
